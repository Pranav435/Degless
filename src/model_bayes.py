"""NumPyro hierarchical degradation model — the core of degless.

Structure, for lap *i* with driver *d*, compound *c*, stint *s*, tyre age *a*
and session-elapsed time *tau*:

    lap_time[i] - evo(tau)  ~  Normal( base[s]
                                       - k_track * burn * lap_in_stint[i]
                                       + deg[c](a)
                                       + dev[d,c] * a ,  sigma_obs )

    deg[c](a) = lin[c] * a  +  hinge[c] * w * softplus( (a - knee[c]) / w )

`evo(tau)` is precomputed by `src.evolution` and enters as data, not as a
parameter.

**Why the softplus hinge.**  A hard `max(0, a - knee)` has a discontinuous
gradient at the knee, which NUTS handles badly.  The softplus is a smooth
approximation with a fixed width `w = 1.5` laps, and it keeps `knee[c]` a
learnable parameter with a real posterior — "cliff at lap 21 +/- 2" is a
statement the model can actually make.

**Two channels.**  Fuel mass slows the car roughly uniformly and dominates
accelerating zones; grip loss hits apex minimum speeds disproportionately.  Two
observables with *different sensitivity signatures* to the same two latent
causes make the fuel/age split identifiable — something a single lap-time
channel mathematically cannot do.  Apex speeds are also robust to practice
engine modes, which wreck lap times but barely touch minimum corner speeds.

**Parameterisation note.**  The plan writes `base[s] ~ Normal(median_lap, 1.5)`.
A free level per stint absorbs the unknown practice fuel load — which is the
point — but it also absorbs the *compound pace offset*, and the strategy
simulator needs that offset (it has to know a SOFT is quicker than a HARD at
equal age).  So `base[s]` is given a structured mean,

    base[s] = driver_base[d] + comp_offset[c] + stint_dev[s]

Degradation is carried by `deg_loss[c](a) = deg[c](a) - deg[c](0)`, the loss
relative to a fresh tyre, so the curve level never fights the intercepts.

**The compound ladder.**  Giving `base[s]` that structure is necessary but not
sufficient: with the per-stint level as diffuse as the unknown fuel load
requires, a free `comp_offset[c] ~ Normal(0, 0.5)` is still swallowed whole.
Fitted that way on Barcelona 2026 it returned MEDIUM 0.10 s/lap *quicker* than
SOFT on offsets of ~0.1 s, and free per-compound slopes returned SOFT degrading
*more slowly* than MEDIUM.  Both invert the physics, and together they are why
the strategy optimiser wanted two 23-lap SOFT stints.

So `lin`, `hinge`, `comp_offset` and `knee` are no longer sampled per compound.
They are built by accumulating **strictly positive steps** down a hardness
ladder (see `src.compounds`):

    lin[c]         = lin_soft / prod(1 + deg_gap[i])   for steps i below c
    comp_offset[c] = sum(pace_gap[i])                  for steps i below c
    knee[c]        = knee_soft + sum(knee_gap[i])      for steps i below c

with `deg_gap`, `pace_gap`, `knee_gap` all LogNormal and therefore positive.
A harder compound can then never come out quicker or faster-degrading than a
softer one, whatever the noise does, while the *size* of every step stays free
for the likelihood to set.  This is the same move as the fuel prior — pin the
part the data cannot identify, estimate the part it can — applied to the
between-compound confound instead of the within-stint one.  It also removes
six weakly identified parameters, which is why the sampler behaves better.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.config import (
    HINGE_WIDTH_LAPS,
    DRIVER_BASE_PRIOR_SD,
    MU_LIN_PRIOR_SCALE,
    SIGMA_LIN_PRIOR_SCALE,
    STINT_LEVEL_PRIOR_SD,
    NUTS_CHAINS,
    NUTS_DRAWS,
    NUTS_TARGET_ACCEPT,
    NUTS_WARMUP,
    RHAT_GATE,
    Event,
    get_event,
)
from src.compounds import get_compound_prior, hardness_rank
from src.fuel import get_prior

log = logging.getLogger("degless.bayes")


def _setup_jax(n_chains: int) -> None:
    import numpyro

    numpyro.set_host_device_count(n_chains)
    numpyro.enable_x64(False)


# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------


def _model(
    stint_idx, comp_idx, drv_idx, age, lap_in_stint, y,
    n_stints, n_comp, n_drv, k_hat, k_sd, burn, base_loc, w, use_hinge,
    mu_lin_scale, sigma_lin_scale, stint_sd, driver_sd, knee_high,
    step_mask, pace_step_s, pace_step_ln_sd, deg_ratio, deg_ratio_ln_sd,
    knee_step_laps, knee_step_ln_sd,
    apex_v=None, apex_corner=None, apex_drv=None, apex_comp=None,
    apex_age=None, n_corner=0, n_apex_drv=0,
):
    import jax
    import jax.numpy as jnp
    import numpyro
    import numpyro.distributions as dist
    from jax.nn import softplus

    # -- the physics prior: Move 1 -----------------------------------------
    k_track = numpyro.sample("k_track", dist.LogNormal(jnp.log(k_hat), k_sd))

    # -- the compound ladder: Move 2 ---------------------------------------
    # Everything below is parameterised in strictly positive *steps* between
    # adjacent compounds, accumulated from the softest.  That is what makes the
    # physical ordering structural rather than something the fit has to
    # rediscover from data that cannot support it (see src/compounds.py).  The
    # size of each step is free; only its sign is fixed.
    #
    # `rank` maps compound index -> hardness rank, 0 = softest.  With n_comp
    # compounds there are n_comp - 1 steps, and `step_mat[j, i] = 1` when step
    # `i` lies below compound `j` on the ladder, so a matrix-vector product
    # accumulates the steps without any Python-level indexing gymnastics.
    n_step = max(n_comp - 1, 1)
    step_mat = jnp.asarray(step_mask)  # (n_comp, n_step), precomputed

    # Degradation of the *softest* compound present; every harder compound is
    # this divided by the accumulated ratios, so it is always smaller.
    lin_soft = numpyro.sample("lin_soft", dist.HalfNormal(mu_lin_scale))
    hinge_soft = numpyro.sample("hinge_soft", dist.HalfNormal(0.10))
    with numpyro.plate("deg_step", n_step):
        # ratio - 1 > 0, so the ratio itself is always > 1: a harder compound
        # can never come out degrading faster than a softer one.
        deg_gap = numpyro.sample(
            "deg_gap", dist.LogNormal(jnp.log(max(deg_ratio - 1.0, 1e-3)),
                                      deg_ratio_ln_sd))
    log_slow = step_mat @ jnp.log1p(deg_gap)          # (n_comp,)
    lin = numpyro.deterministic("lin", lin_soft * jnp.exp(-log_slow))
    hinge = numpyro.deterministic("hinge", hinge_soft * jnp.exp(-log_slow))

    # Pace: the softest compound is the reference at 0, every step adds a
    # strictly positive lap-time penalty.  Without this the free per-stint
    # level (which has to absorb the unknown practice fuel load) swallows the
    # compound offset entirely and returns it as noise near zero.
    with numpyro.plate("pace_step", n_step):
        pace_gap = numpyro.sample(
            "pace_gap", dist.LogNormal(jnp.log(max(pace_step_s, 1e-3)),
                                       pace_step_ln_sd))
    comp_offset = numpyro.deterministic("comp_offset", step_mat @ pace_gap)

    # -- the cliff ----------------------------------------------------------
    # The softest compound's knee is squashed into [5, knee_high] — the upper
    # bound is the oldest tyre actually observed, because a cliff beyond the
    # data is not identifiable and an open range lets the knee free-run
    # whenever the hinge is near zero.  Harder compounds add strictly positive
    # lap gaps on top, so their cliffs always arrive later.
    knee_lo = 5.0
    knee_soft_z = numpyro.sample("knee_soft_z", dist.Normal(0.0, 1.0))
    knee_soft = knee_lo + (knee_high - knee_lo) * jax.nn.sigmoid(knee_soft_z)
    with numpyro.plate("knee_step", n_step):
        knee_gap = numpyro.sample(
            "knee_gap", dist.LogNormal(jnp.log(max(knee_step_laps, 1e-3)),
                                       knee_step_ln_sd))
    knee = numpyro.deterministic("knee", knee_soft + step_mat @ knee_gap)

    # -- levels -------------------------------------------------------------
    with numpyro.plate("drv", n_drv):
        driver_base = numpyro.sample(
            "driver_base", dist.Normal(base_loc, driver_sd))
    with numpyro.plate("stint", n_stints):
        stint_dev = numpyro.sample("stint_dev", dist.Normal(0.0, stint_sd))

    # -- per-driver, per-compound tyre-management deviation ----------------
    sigma_dev = numpyro.sample("sigma_dev", dist.HalfNormal(0.02))
    with numpyro.plate("dc_comp", n_comp, dim=-1):
        with numpyro.plate("dc_drv", n_drv, dim=-2):
            dev_raw = numpyro.sample("dev_raw", dist.Normal(0.0, 1.0))
    dev = numpyro.deterministic("dev", sigma_dev * dev_raw)

    sigma_obs = numpyro.sample("sigma_obs", dist.HalfNormal(0.3))

    # -- degradation loss relative to a fresh tyre -------------------------
    def deg_loss(c, a):
        if not use_hinge:
            return lin[c] * a
        at0 = softplus(-knee[c] / w)
        return lin[c] * a + hinge[c] * w * (softplus((a - knee[c]) / w) - at0)

    mu = (
        driver_base[drv_idx]
        + comp_offset[comp_idx]
        + stint_dev[stint_idx]
        - k_track * burn * lap_in_stint
        + deg_loss(comp_idx, age)
        + dev[drv_idx, comp_idx] * age
    )
    numpyro.sample("obs", dist.Normal(mu, sigma_obs), obs=y)

    # -- second channel: corner apex speeds --------------------------------
    if apex_v is not None:
        with numpyro.plate("corner", n_corner):
            lam = numpyro.sample("lam", dist.HalfNormal(20.0))
        with numpyro.plate("v0_corner", n_corner, dim=-1):
            with numpyro.plate("v0_drv", n_apex_drv, dim=-2):
                v0 = numpyro.sample("v0", dist.Normal(150.0, 60.0))
        sigma_apex = numpyro.sample("sigma_apex", dist.HalfNormal(5.0))
        mu_apex = (
            v0[apex_drv, apex_corner]
            - lam[apex_corner] * deg_loss(apex_comp, apex_age)
        )
        numpyro.sample("obs_apex", dist.Normal(mu_apex, sigma_apex), obs=apex_v)


# --------------------------------------------------------------------------
# Fit wrapper
# --------------------------------------------------------------------------


@dataclass
class BayesFit:
    idata: object = None
    posterior: dict = field(default_factory=dict)   # name -> ndarray (draws, ...)
    compounds: list = field(default_factory=list)
    drivers: list = field(default_factory=list)
    corners: list = field(default_factory=list)
    w: float = HINGE_WIDTH_LAPS
    use_hinge: bool = True
    n_laps: int = 0
    n_apex: int = 0
    max_rhat: float = np.nan
    n_divergences: int = 0
    prior_label: str = ""
    compound_prior_label: str = ""
    event_key: str = ""
    converged: bool = False
    diagnostics: pd.DataFrame | None = None

    # -- curves ---------------------------------------------------------

    def deg_loss(self, compound: str, ages: np.ndarray) -> np.ndarray:
        """Posterior degradation loss vs a fresh tyre — shape (draws, len(ages))."""
        j = self.compounds.index(compound)
        a = np.asarray(ages, dtype=float)[None, :]
        lin = self.posterior["lin"][:, j][:, None]
        if not self.use_hinge:
            return lin * a
        hinge = self.posterior["hinge"][:, j][:, None]
        knee = self.posterior["knee"][:, j][:, None]
        w = self.w
        sp = lambda x: np.logaddexp(0.0, x)  # noqa: E731  softplus
        return lin * a + hinge * w * (sp((a - knee) / w) - sp(-knee / w))

    def curve_table(self, ages: np.ndarray, level: float = 0.90) -> pd.DataFrame:
        lo_q, hi_q = (1 - level) / 2, (1 + level) / 2
        rows = []
        for c in self.compounds:
            d = self.deg_loss(c, ages)
            rows.append(pd.DataFrame({
                "compound": c, "tyre_age": ages,
                "mean": d.mean(0),
                "lo": np.quantile(d, lo_q, axis=0),
                "hi": np.quantile(d, hi_q, axis=0),
            }))
        return pd.concat(rows, ignore_index=True)

    def slope_table(self, level: float = 0.90) -> pd.DataFrame:
        """Mean slope over the first 20 laps — comparable to the MixedLM number."""
        ages = np.arange(0, 21, dtype=float)
        lo_q, hi_q = (1 - level) / 2, (1 + level) / 2
        rows = []
        for c in self.compounds:
            d = self.deg_loss(c, ages)
            slope = (d[:, -1] - d[:, 0]) / (ages[-1] - ages[0])
            j = self.compounds.index(c)
            knee = self.posterior["knee"][:, j]
            rows.append({
                "compound": c,
                "slope_s_per_lap": float(slope.mean()),
                "lo": float(np.quantile(slope, lo_q)),
                "hi": float(np.quantile(slope, hi_q)),
                "knee_lap": float(knee.mean()),
                "knee_lo": float(np.quantile(knee, lo_q)),
                "knee_hi": float(np.quantile(knee, hi_q)),
            })
        return pd.DataFrame(rows)

    def predict_stint(self, compound: str, start_age: float,
                      n_laps: int, n_draws: int | None = None) -> np.ndarray:
        """Posterior-predictive stint pace, shape (n_draws, n_laps).

        Returns degradation loss relative to a fresh tyre, per lap of the stint.
        """
        ages = start_age + np.arange(n_laps, dtype=float)
        d = self.deg_loss(compound, ages)
        if n_draws is not None and n_draws < d.shape[0]:
            idx = np.linspace(0, d.shape[0] - 1, n_draws).astype(int)
            d = d[idx]
        return d

    # -- persistence ----------------------------------------------------

    SAVED = ("lin", "hinge", "knee", "comp_offset", "k_track", "sigma_obs",
             "sigma_dev", "lin_soft", "hinge_soft", "deg_gap", "pace_gap")

    def save(self, path) -> "object":
        """Persist the posterior draws the live engine needs (no InferenceData)."""
        import json as _json
        from pathlib import Path as _P

        path = _P(path)
        arrays = {k: np.asarray(self.posterior[k]) for k in self.SAVED if k in self.posterior}
        np.savez_compressed(path, **arrays)
        meta = {"compounds": list(self.compounds), "drivers": list(self.drivers),
                "corners": [int(c) for c in self.corners], "w": float(self.w),
                "use_hinge": bool(self.use_hinge), "n_laps": int(self.n_laps),
                "n_apex": int(self.n_apex), "max_rhat": float(self.max_rhat),
                "n_divergences": int(self.n_divergences), "prior_label": self.prior_label,
                "compound_prior_label": self.compound_prior_label,
                "event_key": self.event_key, "converged": bool(self.converged)}
        path.with_suffix(".json").write_text(_json.dumps(meta, indent=1))
        return path

    @classmethod
    def load(cls, path) -> "BayesFit":
        import json as _json
        from pathlib import Path as _P

        path = _P(path)
        meta = _json.loads(path.with_suffix(".json").read_text())
        with np.load(path) as z:
            post = {k: z[k] for k in z.files}
        return cls(idata=None, posterior=post, **{k: v for k, v in meta.items()})

    @property
    def comp_offset(self) -> dict:
        return {c: float(self.posterior["comp_offset"][:, j].mean())
                for j, c in enumerate(self.compounds)}

    @property
    def k_track(self) -> np.ndarray:
        return self.posterior["k_track"]


def fit_bayes(df: pd.DataFrame, event: Event | str, *, prior: str = "2026",
              compound_prior: str = "ladder",
              pace_step_s: float | None = None,
              apex: pd.DataFrame | None = None,
              use_hinge: bool = True,
              chains: int = NUTS_CHAINS, warmup: int = NUTS_WARMUP,
              draws: int = NUTS_DRAWS, seed: int = 0,
              target_accept: float = NUTS_TARGET_ACCEPT) -> BayesFit:
    """Sample the hierarchical model.  `df` = clean laps with `evo_s` attached."""
    import arviz as az
    import jax
    import numpyro
    from numpyro.infer import MCMC, NUTS

    _setup_jax(chains)
    ev = get_event(event) if isinstance(event, str) else event
    fp = get_prior(ev, prior)
    cp = get_compound_prior(ev, compound_prior, pace_step_s=pace_step_s)

    d = df.dropna(subset=["lap_time_s", "tyre_age", "compound"]).reset_index(drop=True)
    compounds = sorted(d["compound"].unique())
    drivers = sorted(d["driver"].unique())
    stints = sorted(d["stint_uid"].unique())

    c_idx = d["compound"].map({c: i for i, c in enumerate(compounds)}).to_numpy()
    d_idx = d["driver"].map({x: i for i, x in enumerate(drivers)}).to_numpy()
    s_idx = d["stint_uid"].map({x: i for i, x in enumerate(stints)}).to_numpy()

    evo = d["evo_s"].to_numpy(dtype=float) if "evo_s" in d else np.zeros(len(d))
    y = d["lap_time_s"].to_numpy(dtype=float) - evo
    age = d["tyre_age"].to_numpy(dtype=float)
    lis = d["lap_in_stint"].to_numpy(dtype=float)

    # The compound ladder as a lower-triangular accumulator: step_mask[j, i]
    # is 1 when ladder step `i` sits below compound `j`, so `step_mask @ gaps`
    # is the total of the steps between the softest compound and compound `j`.
    rank = hardness_rank(compounds)
    n_step = max(len(compounds) - 1, 1)
    step_mask = np.zeros((len(compounds), n_step), dtype=np.float32)
    for j, r in enumerate(rank):
        step_mask[j, :r] = 1.0

    # A "no prior" run means no fuel term at all: age and fuel stay collinear.
    k_hat = max(fp.k_track_s_per_kg, 1e-6)
    burn = fp.burn_kg_per_lap
    k_sd = fp.rel_sd

    kw = dict(
        stint_idx=s_idx, comp_idx=c_idx, drv_idx=d_idx, age=age,
        lap_in_stint=lis, y=y, n_stints=len(stints), n_comp=len(compounds),
        n_drv=len(drivers), k_hat=k_hat, k_sd=k_sd, burn=burn,
        base_loc=float(np.median(y)), w=HINGE_WIDTH_LAPS,
        use_hinge=use_hinge, mu_lin_scale=MU_LIN_PRIOR_SCALE,
        sigma_lin_scale=SIGMA_LIN_PRIOR_SCALE,
        stint_sd=STINT_LEVEL_PRIOR_SD, driver_sd=DRIVER_BASE_PRIOR_SD,
        knee_high=float(min(40.0, max(12.0, np.max(age)))),
        step_mask=step_mask, pace_step_s=cp.pace_step_s,
        pace_step_ln_sd=cp.pace_step_ln_sd, deg_ratio=cp.deg_ratio,
        deg_ratio_ln_sd=cp.deg_ratio_ln_sd, knee_step_laps=cp.knee_step_laps,
        knee_step_ln_sd=cp.knee_step_ln_sd,
    )

    corners: list = []
    n_apex = 0
    if apex is not None and len(apex):
        a = apex.dropna(subset=["apex_speed", "tyre_age", "compound"]).copy()
        a = a[a["compound"].isin(compounds)]
        a = a[a["driver"].isin(drivers)]
        if len(a):
            corners = sorted(a["corner"].unique())
            kw.update(
                apex_v=a["apex_speed"].to_numpy(dtype=float),
                apex_corner=a["corner"].map({c: i for i, c in enumerate(corners)}).to_numpy(),
                apex_drv=a["driver"].map({x: i for i, x in enumerate(drivers)}).to_numpy(),
                apex_comp=a["compound"].map({c: i for i, c in enumerate(compounds)}).to_numpy(),
                apex_age=a["tyre_age"].to_numpy(dtype=float),
                n_corner=len(corners), n_apex_drv=len(drivers),
            )
            n_apex = len(a)

    kernel = NUTS(_model, target_accept_prob=target_accept)
    mcmc = MCMC(kernel, num_warmup=warmup, num_samples=draws, num_chains=chains,
                progress_bar=False)
    mcmc.run(jax.random.PRNGKey(seed), extra_fields=("diverging",), **kw)

    idata = az.from_numpyro(mcmc)
    post = {k: np.asarray(v).reshape((-1,) + np.asarray(v).shape[2:])
            for k, v in mcmc.get_samples(group_by_chain=True).items()}

    # -- convergence gate ---------------------------------------------------
    watch = [v for v in ["k_track", "lin", "knee", "hinge", "lin_soft",
                         "hinge_soft", "deg_gap", "pace_gap", "knee_gap",
                         "knee_soft_z", "sigma_dev", "sigma_obs", "comp_offset"]
             if v in idata.posterior]
    summ = az.summary(idata, var_names=watch)
    # A parameter with no posterior variance has an undefined r_hat (0/0).
    # `k_track` is exactly that under the "no prior" variant, where the fuel
    # term is identically zero and k_track is a constant carried along for
    # interface compatibility.  Judge convergence on parameters that actually
    # vary.
    varying = summ[summ["sd"] > 1e-8]
    max_rhat = float(np.nanmax(varying["r_hat"].to_numpy())) if len(varying) else 1.0
    extra = mcmc.get_extra_fields(group_by_chain=False)
    n_div = int(np.sum(np.asarray(extra["diverging"]))) if "diverging" in extra else 0

    fit = BayesFit(
        idata=idata, posterior=post, compounds=compounds, drivers=drivers,
        corners=corners, n_laps=len(d), n_apex=n_apex, max_rhat=max_rhat,
        use_hinge=use_hinge,
        n_divergences=n_div, prior_label=fp.label,
        compound_prior_label=cp.label, event_key=ev.key,
        converged=(max_rhat < RHAT_GATE and n_div == 0),
        diagnostics=summ.reset_index().rename(columns={"index": "param"}),
    )
    log.info("bayes[%s/%s]: %d laps, %d apex rows, max r_hat=%.4f, divergences=%d",
             ev.key, fp.label, fit.n_laps, fit.n_apex, max_rhat, n_div)
    return fit
