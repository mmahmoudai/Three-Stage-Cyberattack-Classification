# Three-Stage Cyberattack Classification

This repository contains code and experiments for three-stage cyberattack classification.

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
