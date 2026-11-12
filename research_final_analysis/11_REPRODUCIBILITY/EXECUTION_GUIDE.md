# Execution guide

```bash
python3 -m venv .venv && ./.venv/bin/pip install -r requirements.txt pytest fpdf2 tabulate
cd research_final_analysis/scripts
../../.venv/bin/python rfa01_datasets.py
../../.venv/bin/python rfa02_implementations.py
../../.venv/bin/python rfa03_ablation.py
../../.venv/bin/python rfa04_tables.py
../../.venv/bin/python rfa05_figures.py
../../.venv/bin/python rfa06_docs.py
../../.venv/bin/python rfa07_manuscript.py
../../.venv/bin/python rfa08_final.py
```
Deterministic given seeds; all inputs are repository bytes or stored phase-5/6 prediction files. Frozen phases are imported, not modified.
