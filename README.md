# Scanned Document Classifier (HITL Project 2)

Semester project for **Human-in-the-Loop Machine Learning** (HSBI, WiSe 2026/27, Dr.-Ing. Marcos Baez).

We build a **human-assisted document classifier**: given the OCR text of a scanned business page, the model predicts its document type and helps decide which pages a person should label or review. The focus is not on the most accurate model but on how humans and the model should work together.

- **Data:** RVL-CDIP (via the course copy `abarsky/rvl_cdip_small`), 8 classes: advertisement, email, form, invoice, letter, news article, presentation, resume
- **Sample:** 300 pages per class (2,400 pages): 1,924 train / 235 validation / 241 test
- **Model:** OCR text (Tesseract) → TF-IDF → logistic regression (final: character n-grams)

## Team

- AshishKumar Maurya
- Yash Desai

## Deliverables

| # | Topic | Main notebook | Status |
|---|---|---|---|
| 0 | Baseline | [Baseline_01.ipynb](Baseline_01.ipynb) | done |
| 1 | Active learning | [active_learning.ipynb](active_learning.ipynb) | done (submitted for Deliverable 1) |

## Deliverable 1: Active learning

**Main notebook:** [active_learning.ipynb](active_learning.ipynb)

- **Setup:** 160 initial labels (20 per class), 10 rounds × 40 pages (budget 400 → 560 labels), simulated oracle, character n-gram TF-IDF + logistic regression, validation macro-F1, 5 seeds
- **Strategies:** random, uncertainty (least confidence), diversity (farthest-first), representativeness (relative representativeness)

| Strategy | Macro-F1 after 560 labels (mean ± std, 5 seeds) | Beats random (seeds) |
|---|---|---|
| Least confidence | 0.682 ± 0.017 | 4 / 5 |
| Random | 0.671 ± 0.017 | – |
| Farthest-first | 0.659 ± 0.023 | 2 / 5 |
| Representative | 0.651 ± 0.026 | 2 / 5 |
| *All 1,924 labels (reference)* | *0.726* | |

**Conclusion:** uncertainty sampling is slightly better than random only late in the process, and the gain is within the run-to-run variability. Diversity and representative sampling are worse than random because they select pages with empty or very short OCR text. For this task, handling pages without usable text matters more than the acquisition strategy.

## Baseline results

| Model | Validation macro-F1 | Test macro-F1 |
|---|---|---|
| Majority class (trivial) | 0.027 | – |
| Word n-grams (starter baseline) | 0.682 | 0.732 |
| Character n-grams (final) | 0.726 | 0.758 |

Main findings: OCR text amount varies strongly by class; *presentation* is the weakest class (ambiguous label); picture-like pages (e.g. advertisements) contain little text; the model is less confident on wrong predictions.

## How to run

The notebooks were run on the yourAI cluster using the course environment from the course repository [mbaezpy/hsbi-hitl-2026](https://github.com/mbaezpy/hsbi-hitl-2026).

### 1. Environment

```bash
git clone https://github.com/mbaezpy/hsbi-hitl-2026.git
cd hsbi-hitl-2026
# use the course environment (on the cluster: module load miniconda3 && conda activate hitl-course)
sudo apt-get install tesseract-ocr tesseract-ocr-eng   # OCR engine (already installed on the cluster)
python -m pip install pytesseract
```

Main Python packages: pandas, scikit-learn, matplotlib, pillow, pytesseract.

### 2. Data (not included in this repository)

```bash
# download the course copy of RVL-CDIP (40,000 pages)
python scripts/download_dataset.py "abarsky/rvl_cdip_small" "data/rvl_cdip_small"

# sample 300 pages per class and run OCR (about 1 hour)
python project_baselines/project_02/scripts/prepare_rvl_cdip_small.py --examples-per-class 300 --seed 42 --tesseract-lang eng --output project_baselines/project_02/data/prepared/rvl_cdip_small_300
```

This creates `project_baselines/project_02/data/prepared/rvl_cdip_small_300/dataset.csv` (OCR text, label, split) and the page images.

### 3. Notebooks

Place the notebooks in `hsbi-hitl-2026/project_baselines/project_02/` and run them with the course kernel (Kernel → Restart and Run All).

- `Baseline_01.ipynb`: data exploration, baseline evaluation, error analysis, character n-gram improvement
- `active_learning.ipynb`: active-learning experiment (the full experiment with 5 seeds takes about 9 minutes on the cluster)
