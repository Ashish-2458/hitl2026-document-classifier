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

## Repository structure

```text
hitl2026-document-classifier/
├── README.md
├── Baseline_01.ipynb          # baseline: exploration, evaluation, error analysis, improvement
├── active_learning.ipynb      # Deliverable 1: active-learning experiment
└── data/prepared/rvl_cdip_small_300/
    └── dataset.csv            # OCR text, label and split of our 2,400 pages
```

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

### Quick start: run the active-learning experiment (about 10 minutes)

This repository includes our prepared data (`data/prepared/rvl_cdip_small_300/dataset.csv`, 2.6 MB: OCR text, label and split of the 2,400 pages). The active-learning notebook only needs this file, so no dataset download and no OCR are required.

**Requirements:** Python 3.10 or newer.

```bash
git clone https://github.com/Ashish-2458/hitl2026-document-classifier.git
cd hitl2026-document-classifier
pip install pandas scikit-learn matplotlib jupyter
jupyter lab
```

Open `active_learning.ipynb` and run Kernel → Restart and Run All. The data is the same as ours; results should match closely (a different scikit-learn version may change the last decimals).

### Full setup: recreate the data from scratch

The baseline notebook also displays page images, which are not included here. To run it, or to recreate the data from scratch, follow the steps below.

Our notebooks use the data-preparation script and the conda environment from the course repository [mbaezpy/hsbi-hitl-2026](https://github.com/mbaezpy/hsbi-hitl-2026). We ran everything on the yourAI cluster; the steps below also work on a local machine with Linux, macOS or Windows (WSL or Git Bash).

**Requirements:** Git, Miniconda/Anaconda, Tesseract OCR, about 5 GB of free disk space.

#### 1. Get both repositories

```bash
git clone https://github.com/mbaezpy/hsbi-hitl-2026.git
git clone https://github.com/Ashish-2458/hitl2026-document-classifier.git

# copy our notebooks next to the project starter code
cp hitl2026-document-classifier/*.ipynb hsbi-hitl-2026/project_baselines/project_02/
cd hsbi-hitl-2026
```

#### 2. Environment

```bash
# creates the conda environment "hitl-course" and a Jupyter kernel "HITL Course"
bash scripts/install_env.sh
conda activate hitl-course
python -m pip install pytesseract
```

Install the Tesseract OCR engine (with English language data):
- Ubuntu/Debian: `sudo apt-get install tesseract-ocr tesseract-ocr-eng`
- macOS: `brew install tesseract`
- Windows: installer from [UB Mannheim Tesseract](https://github.com/UB-Mannheim/tesseract/wiki), then add it to `PATH`
- yourAI cluster: already installed (`module load miniconda3 && conda activate hitl-course`)

#### 3. Data (full dataset and images, not included in this repository)

```bash
# download the course copy of RVL-CDIP (40,000 pages)
python scripts/download_dataset.py "abarsky/rvl_cdip_small" "data/rvl_cdip_small"

# sample 300 pages per class and run OCR (about 1 hour)
python project_baselines/project_02/scripts/prepare_rvl_cdip_small.py --examples-per-class 300 --seed 42 --tesseract-lang eng --output project_baselines/project_02/data/prepared/rvl_cdip_small_300
```

This creates `project_baselines/project_02/data/prepared/rvl_cdip_small_300/dataset.csv` (OCR text, label, split) and the page images.

#### 4. Run the notebooks

```bash
cd project_baselines/project_02
jupyter lab
```

Open the notebooks, select the kernel **HITL Course**, and run Kernel → Restart and Run All.

- `Baseline_01.ipynb`: data exploration, baseline evaluation, error analysis, character n-gram improvement
- `active_learning.ipynb`: active-learning experiment (5 seeds × 4 strategies, about 9 minutes on the cluster)

**Note:** the sampling and the experiments use fixed seeds. A different Tesseract version can produce slightly different OCR text, so numbers may differ slightly from the outputs saved in the notebooks.
