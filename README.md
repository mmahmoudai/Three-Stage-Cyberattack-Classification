# Three-Stage Cyberattack Classification

This repository contains code and experiments for three-stage cyberattack classification.

## Introduction

This project implements a three-stage hierarchical ensemble framework for holistic DDoS attack classification:

- **Stage 1**: Binary traffic screening (Benign vs Attack).
- **Stage 2**: Fine-grained multi-class attack classification for samples predicted as Attack.
- **Stage 3**: Verification/correction logic to reduce cascading errors and improve overall Macro-F1.

The repository includes experiment runners (`run_experiment.py` and `run_experiment_v4.py`) and supporting modules under `src/`.

## License & Citation

This repository is **not** provided under an open-source license. You must obtain permission from the authors/maintainers before using, copying, modifying, or distributing this code.

If you use this code in academic work, you must cite the associated publication and include an appropriate reference.

- **Contact (license requests / permissions)**:
  - Mohammed A. Farsi: mafarsi@taibahu.edu.sa
  - AbdelMoniem M. Helmy: abdelmoniem.hafez@cu.edu.eg
  - Muhammad M. Mahmoud: m.mahmoud@mau.edu.eg
- **Preferred citation**:
  - Mohammed A. Farsi, AbdelMoniem M. Helmy, and Muhammad M. Mahmoud, "Three-Stage Hierarchical Ensemble Framework for Holistic DDoS Attack Classification".

## Dataset

This project expects a CSV dataset at:

```text
data/cicddos2019_dataset.csv
```

The `data/` directory is intentionally **not** committed to GitHub.

### How to get the dataset

Obtain the **CICDDoS2019** dataset from the Canadian Institute for Cybersecurity (CIC) / University of New Brunswick (UNB) distribution (or a mirrored source you trust). Convert/prepare it as a single CSV and place it at `data/cicddos2019_dataset.csv`.

Your CSV must include (at minimum) these columns (see `configs/default.json`):

- `Protocol`
- `Class`
- `Category`

If you place the CSV somewhere else, you can override the path when running:

```bash
python run_experiment.py --csv_path path/to/your_dataset.csv
```

or for V4:

```bash
python run_experiment_v4.py --data path/to/your_dataset.csv
```

## Setup

1. Create and activate a Python environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Running

- Run the main experiment script:

```bash
python run_experiment.py
```

- Alternative experiment version:

```bash
python run_experiment_v4.py
```
