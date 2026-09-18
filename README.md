# Advanced Deep Learning — Assignments

Coursework for Advanced Deep Learning. Each assignment lives in its own
folder and is runnable independently from the terminal. All code runs on
CPU-only PyTorch.

## Repo structure

```
advanced-deep-learning-assignments/
├── environment.yml              # conda env (recommended)
├── requirements.txt             # pip alternative
├── .gitignore
├── assignment-01-denoising-colorization/
│   ├── README.md                # what this assignment does + how to run it
│   ├── main.py                  # entry point (run this)
│   ├── model.py                 # network architecture(s)
│   ├── dataset.py                # dataset / data-loading / noise logic
│   ├── utils.py                  # training loop, visualization helpers
│   └── data/                    # downloaded datasets (gitignored)
├── assignment-02-.../
│   └── ...
└── assignment-03-.../
    └── ...
```

Each assignment folder is self-contained: `cd` into it and run `python main.py`.

## One-time setup

```bash
git clone https://github.com/<your-username>/advanced-deep-learning-assignments.git
cd advanced-deep-learning-assignments

conda env create -f environment.yml
conda activate adl-cpu
```

(If you'd rather use pip: `pip install -r requirements.txt` inside your own venv.)

## Running an assignment

```bash
cd assignment-01-denoising-colorization
conda activate adl-cpu        # if not already active
python main.py
```

Each assignment's own README has the specifics (what it trains, expected
runtime on CPU, output files).

## Adding a new assignment

```bash
mkdir assignment-0N-short-name
cd assignment-0N-short-name
# main.py, model.py, dataset.py, utils.py, README.md as needed
```

Commit as usual:

```bash
git add assignment-0N-short-name
git commit -m "Add assignment N: <short description>"
git push
```
