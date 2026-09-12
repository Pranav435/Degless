VENV := .venv/bin
EVENT ?= italy-2026

.PHONY: help run cache history weekend live live-static replay postrace app test verify clean-processed outlook recalibrate benchmark

help:             ## list targets
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-16s %s\n", $$1, $$2}'

cache:            ## pre-cache every dry 2026 weekend incl. telemetry (long pole)
	$(VENV)/python scripts/00_cache.py --events australia-2026 japan-2026 barcelona-2026 austria-2026 belgium-2026 hungary-2026 $(EVENT)

history:          ## full retrospective on every scored weekend: fit all, recalibrate leave-one-out, decide all
	for e in australia-2026 japan-2026 barcelona-2026 austria-2026 belgium-2026 hungary-2026 italy-2026; do \
	  $(VENV)/python scripts/10_pipeline.py --event $$e --stage fit --boot 50; done
	$(VENV)/python scripts/80_recalibrate.py
	for e in australia-2026 japan-2026 barcelona-2026 austria-2026 belgium-2026 hungary-2026 italy-2026; do \
	  $(VENV)/python scripts/10_pipeline.py --event $$e --stage decide; done

recalibrate:      ## leave-one-out recalibration of the plan-deciding constants -> data/processed/calibration.json
	$(VENV)/python scripts/80_recalibrate.py

benchmark:        ## the benchmark suite (bench/run_all.sh) -> bench/out/
	zsh bench/run_all.sh

pipeline:         ## full retrospective pipeline on one weekend (needs its race)
	$(VENV)/python scripts/10_pipeline.py --event $(EVENT)

weekend:          ## pre-race model: fit on the practice run so far, seal, posterior for the live engine
	$(VENV)/python scripts/40_weekend.py --event $(EVENT)

live:             ## live daemon on the official feed (F1TV token used if FastF1 has one)
	$(VENV)/python scripts/50_live.py --event $(EVENT)

live-static:      ## live daemon on the free archive-polling fallback
	$(VENV)/python scripts/50_live.py --event $(EVENT) --source static

replay:           ## replay an archived race at 20x through the live path (DIR=data/raw/livetiming/2026_hungary_race)
	$(VENV)/python scripts/50_live.py --event $(EVENT) --source recorded --session-dir $(DIR) --speed 20

postrace:         ## after the flag: score the sealed model and the live engine's calls
	$(VENV)/python scripts/60_postrace.py --event $(EVENT)

app:              ## Streamlit dashboard (Live tab first)
	$(VENV)/streamlit run app/dashboard.py

login:            ## F1TV sign-in for car telemetry (make run does this by itself; PASTE=1 to paste the cookie, STATUS=1 to just check)
	$(VENV)/python scripts/f1login.py $(if $(PASTE),--paste,) $(if $(STATUS),--status,)

test:             ## unit + replay tests
	$(VENV)/python -m pytest tests -q

verify:           ## the plan's verification assertions on the scored weekends
	$(VENV)/python scripts/30_verify.py

clean-processed:
	rm -f data/processed/*.parquet data/processed/*.json data/processed/*.npz

run:              ## THE command: dashboard + calendar-driven feed, refits, scoring and F1TV login, all automatic
	$(VENV)/python scripts/run.py $(if $(REHEARSE),--rehearse $(REHEARSE),) $(if $(NOLOGIN),--no-login,)

outlook:          ## the outlook for a weekend from everything known so far (EVENT=..., SESSION=<live practice key> to fold the live board in)
	$(VENV)/python scripts/70_outlook.py --event $(EVENT) $(if $(SESSION),--session $(SESSION),)
