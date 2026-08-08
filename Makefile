PYTHON ?= python3
PYTHONPATH := $(CURDIR)/src
M1_EVIDENCE_DIR ?= evidence/raw
M1_VISUAL_DIR ?= artifacts/m1-atlas
export PYTHONPATH

.PHONY: doctor check repository-projection-check evidence-m0 evidence-m1 evidence-m1-check evidence-m1-verify visuals-m1 visuals-m1-test test demo report clean

doctor:
	$(PYTHON) -m stratafold doctor --workspace "$(CURDIR)"

check: doctor
	$(PYTHON) scripts/check_repo.py

repository-projection-check:
	$(PYTHON) scripts/check_repository_projection.py --json

evidence-m0: doctor
	$(PYTHON) scripts/verify_m0.py --output evidence/raw/m0_gate.json

evidence-m1: doctor
	$(PYTHON) scripts/capture_m1.py --output-dir "$(M1_EVIDENCE_DIR)"

evidence-m1-check: doctor
	$(PYTHON) scripts/capture_m1.py --output-dir "$(M1_EVIDENCE_DIR)" --check

evidence-m1-verify:
	$(PYTHON) scripts/verify_m1_evidence.py --json

visuals-m1: doctor
	mkdir -p "$(M1_VISUAL_DIR)"
	$(PYTHON) scripts/capture_m1_rejection.py --output-dir "$(M1_VISUAL_DIR)"
	$(PYTHON) scripts/render_m1_atlas.py --output-dir "$(M1_VISUAL_DIR)"

visuals-m1-test:
	$(PYTHON) -m unittest tests.test_m1_visuals -v

test: doctor
	$(PYTHON) -m unittest discover -s tests -v
	$(PYTHON) scripts/check_repo.py
	$(MAKE) repository-projection-check
	$(MAKE) evidence-m1-verify

demo: doctor
	@echo "demo is introduced at the M2 vertical-slice gate"

report: doctor
	@echo "report is introduced after raw M3 evidence exists"

clean:
	@echo "Refusing implicit cleanup: remove only explicitly reviewed generated paths."
