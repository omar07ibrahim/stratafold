PYTHON ?= python3
PYTHONPATH := $(CURDIR)/src
export PYTHONPATH

.PHONY: doctor check evidence-m0 test demo report clean

doctor:
	$(PYTHON) -m stratafold doctor --workspace "$(CURDIR)"

check: doctor
	$(PYTHON) scripts/check_repo.py

evidence-m0: doctor
	$(PYTHON) scripts/verify_m0.py --output evidence/raw/m0_gate.json

test: doctor
	$(PYTHON) -m unittest discover -s tests -v
	$(PYTHON) scripts/check_repo.py

demo: doctor
	@echo "demo is introduced at the M2 vertical-slice gate"

report: doctor
	@echo "report is introduced after raw M3 evidence exists"

clean:
	@echo "Refusing implicit cleanup: remove only explicitly reviewed generated paths."
