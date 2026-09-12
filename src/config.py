"""degless — global configuration.

Every constant that the pipeline depends on lives here so that a reviewer can
audit the physics in one file.  Nothing in this module imports anything heavy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
FASTF1_CACHE = DATA_RAW / "fastf1_cache"
DATA_PROCESSED = ROOT / "data" / "processed"
SEALED_DIR = ROOT / "predictions" / "sealed"

for _p in (FASTF1_CACHE, DATA_PROCESSED, SEALED_DIR):
    _p.mkdir(parents=True, exist_ok=True)


def load_dotenv(path=None) -> dict:
    """Read `KEY=value` lines from `.env` into the environment (never overriding
    a variable that is already set).  Returns what was loaded."""
    import os

    path = Path(path) if path else ROOT / ".env"
    out = {}
    if not path.exists():
        return out
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k, v = k.strip(), v.strip().strip('"').strip("'")
        if k and v and k not in os.environ:
            os.environ[k] = v
            out[k] = v
    return out


load_dotenv()

# --------------------------------------------------------------------------
# 2026 regulations physics
# --------------------------------------------------------------------------

FUEL_ALLOWANCE_2026_KG = 70.0  # race fuel allowance, down from 110 kg
MIN_CAR_WEIGHT_2026_KG = 768.0  # minimum weight incl. driver, down from 800 kg
MID_STINT_FUEL_KG = 35.0  # ~half the allowance, the mass we linearise about

# 2025 comparison prior (for the "use last year's physics" sensitivity slide)
FUEL_BURN_2025_KG_PER_LAP = 1.67
K_TRACK_2025_S_PER_KG = 0.033

# Dimensionless mass sensitivity: a 1% mass increase costs ~ALPHA_MASS% of lap
# time.  0.30 is the standard aero/mechanical rule of thumb.
ALPHA_MASS = 0.30

# Relative SD of the LogNormal prior on k_track.  Widened from the usual ~0.15
# because 2026 is a brand new regulation set.
K_TRACK_PRIOR_REL_SD = 0.25

# Tyre wear is load-sensitive: rubber abrades and heats with the energy going
# through the contact patch, and that energy scales with vertical load.  A car
# carrying a full 70 kg of fuel therefore consumes its tyre faster than the same
# car on fumes, which is part of why real stints get *longer* as a race goes on.
#
# The exponent is a load sensitivity, not a free parameter.  Abrasive wear is
# first-order linear in normal load; the load sensitivity of the friction
# coefficient and the thermal feedback into the carcass add curvature on top,
# which puts the effective exponent in the 1.5-2.0 range.  1.6 is used here.
#
# A previous version of this file carried 5.0, fitted by regressing log stint
# degradation on log car mass within compound and event.  That regression is
# not identified: car mass is a deterministic function of race lap, so it is
# collinear with track evolution (the track rubbers in, worth ~0.08 s/lap of
# lap-time drift measured on both 2026 weekends) and with the convention that
# teams run softer compounds early and harder ones late.  The fit was reading
# those two effects as tyre physics.  At 5.0 the first lap of a race wears the
# tyre 1.24x as fast as mid-race and the last lap 0.80x; at 1.6 the spread is
# 1.07x to 0.93x, which is what a load sensitivity actually supports.
TYRE_LOAD_EXPONENT = 1.6

# --------------------------------------------------------------------------
# Clean-lap rules
# --------------------------------------------------------------------------

MIN_STINT_LAPS = 6  # long runs only; excludes quali simulations
TRAFFIC_GAP_S = 2.0  # gap to car ahead below which a lap is dirty
SLOW_LAP_MARGIN_S = 5.0  # lap_time > session_median + this  -> drop
VALID_COMPOUNDS = ("SOFT", "MEDIUM", "HARD")
GREEN_FLAG = "1"

# --------------------------------------------------------------------------
# Telemetry / apex speeds
# --------------------------------------------------------------------------

CORNER_WINDOW_M = 60.0  # +/- metres around the corner marker
SLOW_CORNER_FRAC = 0.6  # median apex < this * session max speed => "slow corner"
N_APEX_CORNERS = 4  # keep the 4 highest-variance slow corners

# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------

HINGE_WIDTH_LAPS = 1.5  # softplus smoothness `w`, fixed for sampler friendliness
# Prior scale on the grid-level linear degradation slope.  The plan proposes
# HalfNormal(0.06); on Barcelona (a high-degradation circuit) the data pull the
# posterior to ~0.13 s/lap, 3.7 sigma into that prior's tail, so it fights the
# likelihood and shrinks every compound slope downward -- visible as a ~0.04
# s/lap disagreement with the MixedLM baseline.  Widened so that the pooling
# across compounds comes from `sigma_lin` (which is what the plan wants it to
# come from) rather than from the prior's mode sitting at zero.
MU_LIN_PRIOR_SCALE = 1.0
SIGMA_LIN_PRIOR_SCALE = 0.08  # between-compound spread; 0.03 was similarly tight

# --------------------------------------------------------------------------
# Compound ladder — the second unidentifiable half, pinned like the fuel one
# --------------------------------------------------------------------------
#
# Practice cannot resolve how the compounds rank against each other.  A long
# run's level is set by an unknown fuel load and engine mode, so a free
# per-stint intercept absorbs the entire compound *pace* difference; and the
# compounds are not run in comparable conditions (at Barcelona 2026 the SOFT
# appears in 96 clean laps of short early runs and the HARD in 9), so the
# per-compound *slopes* come out in whatever order the noise dictates.  Fitted
# freely, Barcelona 2026 produced SOFT degrading more slowly than MEDIUM and
# MEDIUM *faster on pace* than SOFT — both physically impossible, and together
# they are why the strategy optimiser wanted two 23-lap SOFT stints.
#
# So the ladder is pinned by construction, exactly as the fuel effect is:
# softer is always quicker and always degrades faster, with the *size* of each
# step left to the data through an informative prior rather than fixed.

COMPOUND_ORDER = ("SOFT", "MEDIUM", "HARD")  # softest first

# Pace step between adjacent compounds, as a fraction of lap time, used when no
# donor weekend is available to measure it (see `compounds.measure_pace_step`).
#
# Pirelli's design target is ~0.6-0.8 s per step, which would be ~0.8% of lap
# time.  Measured on these 2026 cars it is far smaller: 0.224 +/- 0.090 s at
# Barcelona and 0.117 +/- 0.088 s pooled across both weekends, i.e. 0.15-0.29%
# of lap time.  0.25% sits between the two estimates, and the LogNormal scale
# is wide enough to cover both.
#
# The size of this number decides the compound recommendation outright.  At
# 0.8% the optimiser puts the race on SOFTs, because a 1.25 s/lap penalty for
# the HARD outweighs its far lower wear; at 0.25% it puts the long stints on
# HARDs, which is what every front-running car actually did at Barcelona 2026.
COMPOUND_PACE_STEP_FRAC = 0.0021
COMPOUND_PACE_STEP_REL_SD = 0.35  # LogNormal scale: a factor ~1.4 either way

# 0.21% of lap time, ~0.16 s per step.  This number is the one that decides the
# compound recommendation outright, and getting it from the obvious place gets
# it wrong.
#
# There are two different quantities here and only one of them is what a
# strategy needs:
#
#   (a) pace at *equal tyre age* - how much faster a fresh SOFT is than a fresh
#       HARD.  Estimated with driver and race-lap fixed effects on Barcelona
#       2026 race laps this is +0.35 +/- 0.07 s per step, and it is real.
#
#   (b) pace over a *whole stint*, at the length each compound is actually run
#       to.  Measured on stint means, holding stint length and race phase
#       fixed: +0.027 s/step at Barcelona and +0.042 at Hungary.  Essentially
#       zero.  The SOFT's advantage when fresh is given back through faster
#       degradation before the stint is over.
#
# Both are true.  A model that charges (a) on every lap of a stint *and* prices
# degradation separately counts the SOFT's advantage twice, and the SOFT then
# dominates at every stint length - which is how this file briefly ended up at
# 0.45%, and with it a recommendation that used no HARD at all at two circuits
# where the field ran 48% and 63% of its race laps on the HARD.
#
# So the fresh-tyre step is calibrated *conditional on the degradation ladder*,
# by requiring the two together to reproduce (b).  Solved on each weekend
# separately - back out what this model's degradation curve already charges
# over a stint of the length it recommends, then ask what fresh-tyre step
# leaves the net matching the measurement:
#
#     Barcelona 2026   deg charges 0.174/step over 16 laps -> 0.164 s = 0.210%
#     Hungary 2026     deg charges 0.094/step over 23 laps -> 0.154 s = 0.200%
#
# The two circuits agree to within 0.01 s despite a 3.5x difference in absolute
# degradation, which is the evidence that a constant *fraction of lap time* is
# the right form and that no circuit-severity scaling is needed.
#
# The deeper point: the pace step and the degradation ladder are not separately
# identified by this data, but their *combination* at the stint level is, and
# it is that combination the optimiser consumes.  Calibrating either one alone
# is how a model ends up confidently recommending a tyre nobody ran.

# Degradation ratio between adjacent compounds (softer / harder).  More grip
# means more energy into the tyre, so wear rises with softness.
#
# 1.30, not the 1.8 this file used to carry.  1.8 came from Barcelona 2026 race
# stints measured with a within-stint estimator - SOFT/MEDIUM 1.43, MEDIUM/HARD
# 2.34, geometric mean 1.83 - and that estimator is biased.  Within a stint,
# race lap and tyre age advance together, so it absorbs the whole track
# evolution drift (~-0.087 s/lap at Barcelona) into the age slope.  The bias is
# a roughly constant subtraction from every compound's rate, so it inflates
# *ratios* between them the smaller the rate: it takes the HARD from 0.110 to
# 0.040 s/lap and the SOFT from 0.186 to 0.136, and 0.136/0.040 is a much
# bigger number than 0.186/0.110.
#
# Re-measured with driver and race-lap fixed effects plus a dirty-air control:
# SOFT 0.186, MEDIUM 0.155, HARD 0.110 s/lap, giving SOFT/MEDIUM 1.20 and
# MEDIUM/HARD 1.41, geometric mean 1.30.  The ordering is intact - softer still
# degrades faster - but the spread is less than half what was assumed.
COMPOUND_DEG_RATIO = 1.30
COMPOUND_DEG_RATIO_LN_SD = 0.35

# Cliff ordering: a softer compound reaches its knee earlier, in laps.
COMPOUND_KNEE_STEP_LAPS = 3.0
COMPOUND_KNEE_STEP_LN_SD = 0.5

# --------------------------------------------------------------------------
# Practice -> race regime transfer
# --------------------------------------------------------------------------
#
# A practice long run and a race stint are not the same experiment.  In a long
# run the driver pushes every lap on a hot afternoon track to gather data; in
# the race the same driver lifts and coasts, manages temperatures, runs in
# dirty air and is on a fuel and engine-mode plan.  Measured on Barcelona 2026,
# race stints degrade at 0.37x the practice rate.  Ignoring this is the single
# largest error in the strategy math: it triples the cumulative cost of a long
# stint, and the optimiser answers by pitting as early as the rules allow.
#
# The factor is *measured from other weekends' races*, never from the target
# weekend's — that would breach the practice-only firewall.  This default is
# the fallback for the first weekend of a season, when no donor race exists.
# 0.40 is the one measurement this project has (Barcelona 2026 pools to 0.37),
# rounded toward 1 so the default is the conservative side of the evidence.
# The default is now only a *fallback and a cross-check*, not a multiplier the
# strategy math depends on: `src.tyre` derives the effective regime factor from
# the push level the optimiser chooses, and `src.regime` measures one so the two
# can be compared.  It is set to 0.65 rather than the old 0.40 because the old
# number was produced by a broken estimator.
#
# The bug: `regime.py` removed track evolution from the *practice* side and not
# from the *race* side, then divided one by the other.  Within a stint, race lap
# and tyre age advance together, so the race-side slope absorbed the full track
# evolution drift - measured at -0.087 s/lap (Barcelona) and -0.079 s/lap
# (Hungary) after fuel correction - and came out biased low by roughly that
# much on every compound.  At Hungary it drove the measured race degradation
# *negative* (SOFT -0.022 s/lap), which is not a thing tyres do.
#
# Corrected, with driver and race-lap fixed effects on both sides:
#
#                       practice      race       ratio
#     Barcelona 2026    0.26 s/lap    0.150      0.57
#     Hungary 2026      0.074         0.065      0.88
#
# 0.65 is between them.  The spread between 0.57 and 0.88 is itself the point:
# this was never a transferable constant, and the old 0.50 ln-SD understated how
# badly it travels.  The model's own validation showed the consequence - against
# the actual races it under-predicted degradation by 1.3x at Barcelona and 3.6x
# at Hungary, and an optimiser fed degradation that is 3x too low will always
# answer with too few stops and stints that are too long.
DEFAULT_RACE_REGIME_RATIO = 0.65
DEFAULT_RACE_REGIME_LN_SD = 0.35
REGIME_MIN_PRACTICE_SLOPE = 0.03  # s/lap below which a ratio is uninformative

# Level priors.  The plan writes `base[s] ~ Normal(median_lap, 1.5)` and says it
# exists to absorb "unknown practice fuel load + engine mode".  1.5 s is too
# tight for that job: a practice long run can start on anything from 0 to 70+ kg
# of fuel, worth up to ~2 s of lap time on its own, with engine modes worth
# another 1-2 s on top.  Measured effect of the tight prior on Barcelona: it
# suppresses the MEDIUM slope from 0.284 to 0.238 s/lap and puts the posterior
# 0.08 s/lap below the MixedLM baseline, i.e. it breaks the cross-check.  A
# diffuse stint level restores agreement, and the car-to-car spread that a
# *driver* level has to carry really is small, so that one stays tight.
STINT_LEVEL_PRIOR_SD = 10.0
DRIVER_BASE_PRIOR_SD = 0.5
NUTS_CHAINS = 4
NUTS_WARMUP = 1500
NUTS_DRAWS = 1500
# 0.97 rather than the plan's 0.90: when `hinge[c]` is near zero the matching
# `knee[c]` is unidentified and the sampler free-runs across its prior, a funnel
# that produces divergences at 0.90 and still a couple at 0.95.  The funnel is
# intrinsic - practice long runs end before any compound reaches its cliff, so
# the hinge really is unidentified - and the strategy model no longer depends
# on that parameter (tyre life comes from the grip budget instead, see
# `src.tyre`), but the fit still has to sample it honestly.
NUTS_TARGET_ACCEPT = 0.98
RHAT_GATE = 1.01
BOOTSTRAP_N = 200  # block bootstrap resamples for the MixedLM baseline

# The hinge is unidentified on every scored weekend (the knee posterior equals
# its prior everywhere: practice long runs end before any compound reaches its
# cliff), and it is what let the circuit-history fold-in amplify a near-zero
# post-knee slope 30-80x at Australia and Italy.  The production fit is now
# linear in tyre age; the cliff comes from the circuit's race history (longest
# and p90 stint per compound) and is reported as such.  The hinge survives as
# a diagnostic variant of the retrospective pipeline.
USE_HINGE_DEFAULT = False

# Race-weekend sampler settings.  The quick fit (2 chains x 800 + 800) scores
# within 0.003 s/lap of the production 4 x 1500 + 1500 on every weekend
# benchmarked, in a third of the time; the weekend refit runs it by default.
WEEKEND_NUTS_CHAINS = 2
WEEKEND_NUTS_WARMUP = 800
WEEKEND_NUTS_DRAWS = 800

# --------------------------------------------------------------------------
# Tyre model: the grip budget and the push/wear trade-off  (see src/tyre.py)
# --------------------------------------------------------------------------

# A tyre reaches its cliff after it has surrendered a roughly fixed amount of
# lap time, not after a fixed number of laps.  Measured on Barcelona 2026 race
# stints (degradation rate from a two-way fixed-effects fit, length from the
# longest stint each compound was actually run to):
#
#     SOFT    0.186 s/lap x 21 laps = 3.91 s
#     MEDIUM  0.155 s/lap x 26 laps = 4.03 s
#     HARD    0.110 s/lap x 31 laps = 3.40 s
#
# Three compounds spanning 1.7x in rate and 1.5x in length agree to within
# +/-9%.  That invariant is what lets tyre life be *derived* from degradation
# rate (life = budget / rate) instead of fitted as a free parameter, which is
# how the cliff gets located on a compound that practice never ran near it.
#
# Hungary 2026 does not pin the budget and is not used to: its stints are
# race-limited rather than tyre-limited (a 70-lap race on tyres good for 40+
# laps), so its observed stint lengths are a lower bound on life, not a
# measurement of it.  It is a consistency check - nothing there contradicts
# ~4 s - and no more.
GRIP_BUDGET_S = 3.8
GRIP_BUDGET_REL_SD = 0.20
# 3.8 s is the Barcelona 2026 number.  Across the seven scored weekends the
# race-measured rate times the longest stint each compound was run to comes out
# between 1.9 and 3.8 s, and the model over-stated tyre life on 19 of 19
# compound-weekends with the constant.  The recalibration script now fits it
# per compound from every scored race but the target's (a stint that ends
# before the cliff is a lower bound, so the upper quartile across weekends is
# taken); this constant is the fallback for a season with no scored race.

# Past the cliff, pace loss runs away.  `grip_loss` adds
# `kappa * softmax(w-1)**q` to the linear-in-wear term, so at w = 1.25 the tyre
# is ~1.5x its pre-cliff loss, at w = 1.5 ~3x, at w = 1.75 ~5x.  That is the
# shape of a tyre that is done, and it is what makes an over-long stint
# expensive for a physical reason rather than because a cap forbade it.
CLIFF_SHARPNESS = 8.0
CLIFF_EXPONENT = 2.0
CLIFF_SMOOTH = 0.05  # softplus width on the wear axis, for a differentiable knee

# The push/wear trade-off.  `p = 1` is a practice long run - full attack, which
# is exactly what a long run is for, and therefore the regime the degradation
# curve is fitted in.  `p < 1` is management: lift and coast, short-shift, roll
# speed through the corner instead of attacking the entry.
#
# MANAGE_COST_S is the lap time given up at maximum management.  Team radio and
# the observed practice/race pace gap both put full tyre management at around a
# second a lap; 0.9 s is used here.
#
# MANAGE_WEAR_FLOOR is how far the wear rate can be cut by management alone.
# The energy through the contact patch is bounded below by the energy needed to
# get round the lap at all, so management cannot take wear to zero: 0.45 means
# maximum management roughly halves the wear rate.  Together with a convex
# exponent, these reproduce the ~0.6 practice->race factor that an
# evolution-corrected measurement of Barcelona 2026 gives (0.57) - the model
# predicts that number rather than being told it, which is the cross-check the
# old exogenous constant could not offer.
MANAGE_COST_S = 0.9
MANAGE_WEAR_FLOOR = 0.45
MANAGE_WEAR_EXPONENT = 1.7

# Push levels the optimiser searches over.  Coarse on purpose: the cost surface
# in `p` is flat near its optimum, and a finer grid multiplies the search for
# differences well inside the posterior's own width.
PUSH_GRID = (0.55, 0.7, 0.85, 1.0)

# --------------------------------------------------------------------------
# Traffic and safety cars: what a plan pays that is not tyre or pit lane
# --------------------------------------------------------------------------

# Dirty air, measured with driver and race-lap fixed effects on 2026 race laps:
# a car within 3 s of the one ahead loses 0.33 s/lap at Barcelona and 0.70 s/lap
# at Hungary, and within 1 s that grows further.  This is larger than the entire
# compound pace ladder, and the previous version of the strategy objective did
# not price it at all - which is why an extra stop was costed as pit-lane time
# alone, with the laps spent recovering through traffic afterwards free.
DIRTY_AIR_S_PER_LAP = 0.45
DIRTY_AIR_REL_SD = 0.40

# Excess close-following laps caused by one pit stop.  Measured as the rise in
# P(gap to car ahead < 3 s) over the ten laps after a stop, net of a race-phase
# baseline, pooled over both 2026 weekends: +0.11 to +0.16 for about ten laps,
# integrating to ~1.2 lap-equivalents.  At 0.45 s/lap that is ~0.55 s per stop.
TRAFFIC_LAPS_PER_STOP = 1.2

# Safety cars.  A stop taken under a safety car costs roughly 40% of a
# green-flag stop, because the whole field is slowed while the pit lane is not.
# Both 2026 weekends show interruptions (Hungary: 6% of race laps under a
# non-green track status), and a strategy model blind to them systematically
# over-values plans that have already used all their stops.
SC_RATE_PER_LAP = 0.006          # ~1 in 3 races sees one in a 60-70 lap event
SC_PIT_LOSS_FRACTION = 0.40

# --------------------------------------------------------------------------
# Strategy
# --------------------------------------------------------------------------

MC_DRAWS = 500  # posterior draws in the race Monte Carlo
PIT_WINDOW_MARGIN = 6  # shortest stint the search will consider, in laps
MAX_STOPS = 3  # 3 is a real option once degradation is priced correctly

# The opening stint is not like the others.  The car leaves from a standing
# start on tyres warmed only by blankets and a formation lap, with no out-lap to
# switch them on, on a full tank, into the densest traffic of the race - the
# measured probability of running within 3 s of another car is 0.69 on lap 1
# against 0.38 by half distance.  A harder compound is worse at every one of
# those: it needs more energy to reach its working range, so it launches and
# brakes worse on lap 1, and places lost there are paid back over the following
# laps in dirty air at the measured 0.45 s/lap.
#
# Without this the opening compound is decided by the fuel-load wear term alone,
# which is small, and the model treats the whole ordering as a near-tie: the
# best plan for each ordering of the same compounds spans under 2 s.  The field
# is not remotely undecided.  Across both 2026 weekends, **0 of 32 classified
# finishers started on the HARD** and the single most common plan at both was
# M-H-H - medium first, then two long stints on the hard.
#
# 1.0 s per step of hardness, charged once, on the opening stint only.  The
# magnitude is set from that revealed preference rather than measured directly:
# two races give no clean estimate of a lap-1 compound effect, but they give an
# unambiguous ranking, and the penalty has to exceed the ~2 s spread across
# orderings for the model to reproduce it.  It is reported in the output as a
# calibrated preference, not a measurement.
GRID_START_PENALTY_S = 1.0

# Cold-tyre cost of the first flying lap of a stint.  Distinct from pit loss,
# which is measured on the in-lap and out-lap themselves.  It is what makes an
# extra stop cost more than just the pit-lane time.
OUT_LAP_PENALTY_S = 0.7

# --------------------------------------------------------------------------
# Track position: the undercut exposure of a stop lap, and the field's plan shapes
# --------------------------------------------------------------------------
#
# The tyre-optimal stop lap is not the lap teams stop on.  Benchmarked on the
# seven dry 2026 weekends, the tyre-only objective called the first stop 3-6
# laps *after* the field's median, because the field is covering the undercut:
# every lap a car stays out past the point where a rival on a fresh tyre would
# gain on it is a lap on which it can lose the place.  The exposure of a stop
# lap is the cumulative undercut gain a rival would have had over the laps the
# car stayed out with the undercut open (see `strategy.undercut_exposure`), and
# it enters the objective at this weight, in seconds per second of exposure,
# scaled by how dense the field is at the stop lap.  Calibrated leave-one-out
# against the field's median first-stop lap (`scripts/80_recalibrate.py`); the
# tyre-optimal plan (lambda = 0) is reported beside the position-aware one.
UNDERCUT_EXPOSURE_LAMBDA = 0.15

# A sequence nobody has run at this circuit needs a large time gain to be
# recommended.  The circuit's historical plan and start-compound frequencies
# enter as a prior over plan families: cost += tau * (-log p(family)), with a
# smoothed frequency that backs off to the start-compound and stop-count
# marginals for a family never seen.  tau is seconds per nat; at 2.5 s a plan
# ten times rarer than the modal one carries a 5.8 s handicap.  Calibrated
# leave-one-out against the share of the field that ran the recommended
# sequence.  Zero switches the prior off.
PLAN_PRIOR_TAU_S = 2.5
PLAN_PRIOR_ALPHA = 2.0      # pseudo-counts of back-off mass in the smoothed frequency

# The circuit's own first-stop history, as a soft prior on the *first* stop lap.
# `src.firststop` turns the circuit's green-flag first stops into a density over
# the lap, and the objective charges `kappa * neglogp[lap]` seconds, zero at the
# modal lap.  kappa is seconds per nat, calibrated leave-one-out against the
# field's median green first stop like lambda and tau.
#
# The default is set from the shape of the density rather than from a fit: the
# circuit-pooled KDE has a +/-4-lap between-year spread, so one lap away from the
# mode costs roughly 0.1-0.2 nats and four laps away 1-1.5 nats.  At kappa = 1 s
# that is a few tenths inside the plausible window - smaller than the ~1 s the
# cost surface itself puts on a one-lap move - and 1-1.5 s at the edge of it,
# which is the weight a prior should carry against a measurement: enough to break
# a near-tie, not enough to override a weekend whose tyres genuinely differ.
# Zero switches the prior off (the V2 objective, and the benchmark's ablation).
FIRST_STOP_KAPPA_S = 1.0

# Per-lap noise of a clean racing lap, used to score the sealed curves against
# race stints.  The practice `sigma_obs` (0.74-1.05 s) describes practice
# laps - engine modes, fuel saving, traffic - and race stints are scored
# centred on their own mean, which removes most of that; 0.5 s is what the
# live engine has used since Barcelona and what the stint-rate intervals need
# to cover 90% without covering 100%.
SIGMA_RACE_LAP_S = 0.5

# A driver cannot run the same compound three times.  Pirelli's dry allocation
# is 13 sets, but most are surrendered or scrubbed through practice and
# qualifying, so what reaches the grid is roughly two usable sets per compound.
# Verified against both weekends: of 44 driver-races, 43 used at most two
# stints on any one compound and exactly one used three.  Without this the
# optimiser happily recommends three or four stints on the softest tyre, which
# is not a strategy anyone can physically execute.
MAX_STINTS_PER_COMPOUND = 2

# How far past the cliff a stint may run before the search stops considering it.
# Expressed in wear, not laps: 1.35 means the tyre may be taken to 135% of its
# grip budget, by which point `grip_loss` has it at more than twice its
# pre-cliff loss rate.  Past that the plan is not one an engineer would call,
# and enumerating it only spends search on answers nobody would give.
#
# Note this is a *search bound*, not the thing that keeps stints sensible.  The
# cliff does that, by making a long stint expensive for a physical reason.  The
# previous version had it the other way round - a flat degradation curve that
# imposed no limit of its own, and a hard cap set at 1.5x the oldest tyre age
# practice happened to reach, which had nothing to do with the tyre and let the
# *least*-supported compound extrapolate furthest.
MAX_WEAR_LIMIT = 1.35

# The other bound on stint length, and a different kind of claim.  The wear
# bound says the tyre is finished; this one says the *curve* has run out of
# evidence.  Degradation is fitted on practice long runs that reach age 15-22
# laps, and a rate extrapolated to a stint twice that long is an assertion, not
# a measurement.  It binds where the wear bound cannot - at a low-degradation
# circuit the grip budget alone permits a stint longer than the race.
SUPPORT_EXTRAPOLATION_LIMIT = 2.0

# --------------------------------------------------------------------------
# Presentation
# --------------------------------------------------------------------------

# Okabe-Ito, colourblind safe.
COMPOUND_COLORS = {
    "SOFT": "#D55E00",
    "MEDIUM": "#F0E442",
    "HARD": "#0072B2",
}

PRACTICE_SESSIONS = ("Practice 1", "Practice 2", "Practice 3")


# --------------------------------------------------------------------------
# Events
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Event:
    """One race weekend.

    `openf1_sessions` are the session keys from the OpenF1 index (loaded from
    the cached `data/raw/openf1_sessions_2026.json`) and drive the OpenF1
    ingest path.  `ff1_gp` / `ff1_round` drive the FastF1 primary path.

    `fit_ok` says whether the weekend has dry, conventional practice running
    the degradation model may be fitted on.  `donor_ok` says whether its race
    may be used as a donor for the transferred quantities (regime factor,
    allocation, pace step, pit loss).  A sprint weekend is a donor but not a
    fit; a wet weekend is neither.
    """

    key: str
    name: str
    ff1_year: int
    ff1_gp: str
    ff1_round: int
    meeting_key: int
    n_race_laps: int
    t_lap_ref_s: float
    openf1_sessions: dict = field(default_factory=dict)
    practice_sessions: tuple = PRACTICE_SESSIONS
    circuit: str = ""
    race_date: str = ""
    sprint: bool = False
    dry: bool = True
    fit_ok: bool = True
    donor_ok: bool = True
    livetiming_path: str = ""
    note: str = ""

    # -- physics derived from the 2026 regulations -------------------------

    @property
    def burn_kg_per_lap(self) -> float:
        """Fuel burned per lap = allowance / race distance."""
        return FUEL_ALLOWANCE_2026_KG / self.n_race_laps

    @property
    def m_total_kg(self) -> float:
        """Mass we linearise the sensitivity about: min weight + mid-stint fuel."""
        return MIN_CAR_WEIGHT_2026_KG + MID_STINT_FUEL_KG

    @property
    def k_track_hat_s_per_kg(self) -> float:
        """Mass sensitivity from first principles: alpha * t_lap / m_total."""
        return ALPHA_MASS * self.t_lap_ref_s / self.m_total_kg

    @property
    def fuel_effect_s_per_lap(self) -> float:
        """Lap-time gain per lap from burning fuel, at the reference mass."""
        return self.k_track_hat_s_per_kg * self.burn_kg_per_lap

    @property
    def is_past(self) -> bool:
        from datetime import date
        return bool(self.race_date) and date.fromisoformat(self.race_date) < date.today()


# --------------------------------------------------------------------------
# The 2026 calendar
# --------------------------------------------------------------------------
#
# One row per weekend.  `n_race_laps` is the scheduled race distance and
# `t_lap_ref_s` the representative quick lap (the 10th percentile of clean
# practice laps where data exist, otherwise a pre-season estimate); both only
# enter through the fuel prior and the compound pace step, where a few percent
# is immaterial.  Weather flags come from the OpenF1 weather feed, surveyed
# session by session on 2026-09-04: a weekend is `dry` only when every practice
# session and the race were rain-free.
#
# Bahrain (round 4 in the original calendar) and Saudi Arabia are missing from
# both the F1 livetiming index and OpenF1 upstream and are not listed.

_CAL = [
    # key, name, round, ff1_gp, meeting, circuit, laps, t_ref, date, sprint, dry, practice, note
    ("australia-2026", "Australia 2026", 1, "Australian Grand Prix", 1279, "Melbourne", 58, 81.5, "2026-03-08", False, True, PRACTICE_SESSIONS, ""),
    ("china-2026", "China 2026", 2, "Chinese Grand Prix", 1280, "Shanghai", 56, 90.0, "2026-03-15", True, True, ("Practice 1",), "sprint weekend: race donor only"),
    ("japan-2026", "Japan 2026", 3, "Japanese Grand Prix", 1281, "Suzuka", 53, 91.8, "2026-03-29", False, True, PRACTICE_SESSIONS, ""),
    ("miami-2026", "Miami 2026", 4, "Miami Grand Prix", 1284, "Miami", 57, 88.0, "2026-05-03", True, False, ("Practice 1",), "sprint weekend; race wet"),
    ("canada-2026", "Canada 2026", 5, "Canadian Grand Prix", 1285, "Montreal", 70, 72.0, "2026-05-24", True, True, ("Practice 1",), "sprint weekend, wet qualifying: race donor only"),
    ("monaco-2026", "Monaco 2026", 6, "Monaco Grand Prix", 1286, "Monte Carlo", 78, 71.0, "2026-06-07", False, False, PRACTICE_SESSIONS, "wet practice; strategy anomaly"),
    ("barcelona-2026", "Barcelona 2026", 7, "Barcelona Grand Prix", 1287, "Barcelona", 66, 78.0, "2026-06-14", False, True, PRACTICE_SESSIONS, ""),
    ("austria-2026", "Austria 2026", 8, "Austrian Grand Prix", 1288, "Spielberg", 71, 68.9, "2026-06-28", False, True, PRACTICE_SESSIONS, ""),
    ("britain-2026", "Britain 2026", 9, "British Grand Prix", 1289, "Silverstone", 52, 88.0, "2026-07-05", True, True, ("Practice 1",), "sprint weekend: race donor only"),
    ("belgium-2026", "Belgium 2026", 10, "Belgian Grand Prix", 1290, "Spa-Francorchamps", 44, 107.2, "2026-07-19", False, True, PRACTICE_SESSIONS, ""),
    ("hungary-2026", "Hungary 2026", 11, "Hungarian Grand Prix", 1291, "Budapest", 70, 77.0, "2026-07-26", False, True, ("Practice 2", "Practice 3"), ""),
    ("netherlands-2026", "Netherlands 2026", 12, "Dutch Grand Prix", 1292, "Zandvoort", 72, 71.0, "2026-08-23", True, False, ("Practice 1",), "wet throughout"),
    ("italy-2026", "Italy 2026", 13, "Italian Grand Prix", 1293, "Monza", 53, 83.0, "2026-09-06", False, True, PRACTICE_SESSIONS, ""),
    ("spain-2026", "Spain 2026", 14, "Spanish Grand Prix", 1294, "Madring", 57, 85.0, "2026-09-13", False, True, PRACTICE_SESSIONS, "new circuit"),
    ("azerbaijan-2026", "Azerbaijan 2026", 15, "Azerbaijan Grand Prix", 1295, "Baku", 51, 102.0, "2026-09-26", False, True, PRACTICE_SESSIONS, ""),
    ("malaysia-2026", "Malaysia 2026", 16, "Malaysian Grand Prix", 1308, "Kuala Lumpur", 56, 92.0, "2026-10-04", False, True, PRACTICE_SESSIONS, "listed as Bahrain in FastF1's schedule"),
    ("singapore-2026", "Singapore 2026", 17, "Singapore Grand Prix", 1296, "Singapore", 62, 92.0, "2026-10-11", True, True, ("Practice 1",), "sprint weekend"),
    ("usa-2026", "USA 2026", 18, "United States Grand Prix", 1297, "Austin", 56, 95.0, "2026-10-25", False, True, PRACTICE_SESSIONS, ""),
    ("mexico-2026", "Mexico 2026", 19, "Mexico City Grand Prix", 1298, "Mexico City", 71, 77.0, "2026-11-01", False, True, PRACTICE_SESSIONS, ""),
    ("brazil-2026", "Brazil 2026", 20, "São Paulo Grand Prix", 1299, "Interlagos", 71, 71.0, "2026-11-08", False, True, PRACTICE_SESSIONS, ""),
    ("lasvegas-2026", "Las Vegas 2026", 21, "Las Vegas Grand Prix", 1300, "Las Vegas", 50, 93.0, "2026-11-21", False, True, PRACTICE_SESSIONS, ""),
    ("qatar-2026", "Qatar 2026", 22, "Qatar Grand Prix", 1301, "Lusail", 57, 82.0, "2026-11-29", False, True, PRACTICE_SESSIONS, ""),
    ("abudhabi-2026", "Abu Dhabi 2026", 23, "Abu Dhabi Grand Prix", 1302, "Yas Marina Circuit", 58, 84.0, "2026-12-06", False, True, PRACTICE_SESSIONS, ""),
]

OPENF1_SESSIONS_FILE = DATA_RAW / "openf1_sessions_2026.json"


def _openf1_keys() -> dict:
    """meeting_key -> {session_name: session_key}, from the cached OpenF1 index."""
    import json

    if not OPENF1_SESSIONS_FILE.exists():
        return {}
    try:
        rows = json.loads(OPENF1_SESSIONS_FILE.read_text())
    except Exception:
        return {}
    out: dict = {}
    for r in rows:
        out.setdefault(int(r["meeting_key"]), {})[r["session_name"]] = int(r["session_key"])
    return out


def _build_events() -> dict:
    keys = _openf1_keys()
    out = {}
    for (key, name, rnd, gp, mk, circuit, laps, tref, date, sprint, dry, prac, note) in _CAL:
        out[key] = Event(
            key=key, name=name, ff1_year=2026, ff1_gp=gp, ff1_round=rnd,
            meeting_key=mk, n_race_laps=laps, t_lap_ref_s=tref,
            openf1_sessions=keys.get(mk, {}), practice_sessions=tuple(prac),
            circuit=circuit, race_date=date, sprint=sprint, dry=dry,
            fit_ok=bool(dry and not sprint), donor_ok=bool(dry), note=note,
        )
    return out


EVENTS: dict[str, Event] = _build_events()

DEV_EVENT = "barcelona-2026"
COLD_EVENT = "hungary-2026"


def get_event(key: str) -> Event:
    if key not in EVENTS:
        raise KeyError(f"unknown event {key!r}; known: {sorted(EVENTS)}")
    return EVENTS[key]


def fit_events() -> list:
    """Weekends the degradation model may be fitted on (dry, conventional)."""
    return [k for k, e in EVENTS.items() if e.fit_ok]


def donor_events(exclude: str | None = None) -> list:
    """Weekends whose race may inform the transferred priors."""
    return [k for k, e in EVENTS.items() if e.donor_ok and k != exclude]


def current_event(today=None) -> Event:
    """The weekend in progress, or the next one on the calendar."""
    from datetime import date, timedelta

    today = today or date.today()
    for e in sorted(EVENTS.values(), key=lambda e: e.race_date):
        if date.fromisoformat(e.race_date) + timedelta(days=0) >= today - timedelta(days=0):
            return e
    return sorted(EVENTS.values(), key=lambda e: e.race_date)[-1]
