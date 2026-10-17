# CS 7830 Assignment 2

Logistic regression from scratch using the Flu dataset.

[![Run assignment](https://github.com/skrakibulislamrahat/CS7830-Assignment-2-Logistic-Regression/actions/workflows/run-analysis.yml/badge.svg)](https://github.com/skrakibulislamrahat/CS7830-Assignment-2-Logistic-Regression/actions/workflows/run-analysis.yml)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/skrakibulislamrahat/CS7830-Assignment-2-Logistic-Regression/blob/main/CS7830_Assignment_2.ipynb)

I completed this project for CS 7830 Machine Learning. The goal was to study the Flu dataset and build logistic regression without using scikit-learn, statsmodels, or another machine-learning toolbox. NumPy, pandas, and Matplotlib are the only third-party libraries used.

## What the project does

- Summarizes the numeric and categorical variables in the dataset.
- Checks correlations and possible redundant features.
- Predicts `Flu` from `Risk` with one-variable logistic regression.
- Fits the required 12-variable model.
- Uses forward selection for a smaller model.
- Tests L2 regularization and feature scaling.
- Compares cross-entropy and mean squared error costs.
- Saves all tables, graphs, and model results inside the `results` folder.

## Main result

Forward selection kept `Risk` and `HndWshQual`. This model had 81.33% test accuracy, compared with 77.33% for the model using all 12 features. Scaling did not change the predictions, but it reduced gradient-descent iterations from 34,416 to 2,384.

| Model | Accuracy | Recall | F1 | AUC |
|---|---:|---:|---:|---:|
| Risk only | 0.8133 | 0.2500 | 0.3636 | 0.7881 |
| All 12 features | 0.7733 | 0.0625 | 0.1053 | 0.7733 |
| Risk + HndWshQual | 0.8133 | 0.1875 | 0.3000 | 0.7341 |

The target is imbalanced, so I did not use accuracy alone when comparing the models.

## Run from the terminal

Download or clone the repository, then run:

```bash
python -m pip install -r requirements.txt
python CS7830_Assignment_2.py
```

The code reads `Flu.csv` from the project folder and creates the complete `results` folder. If the command on the computer is `python3`, use `python3` in both commands.

## Run as a notebook

Open `CS7830_Assignment_2.ipynb` and run the cells from top to bottom. The **Open in Colab** button above also opens a runnable online copy. The notebook calls the same submitted Python file, so both versions use the same calculations.

## Files

| File or folder | Purpose |
|---|---|
| `CS7830_Assignment_2.py` | Complete implementation for Questions 1–4 |
| `CS7830_Assignment_2.ipynb` | Notebook version for running and viewing results |
| `Flu.csv` | Dataset |
| `Assignment_2_Report.pdf` | Written report |
| `requirements.txt` | Required Python packages |
| `results/figures` | Generated graphs |
| `results/tables` | Generated numerical tables |
| `results/summary.json` | Main results in one file |

## Reproducibility

I used random seed 42, a manually stratified 80/20 train-test split, and a classification threshold of 0.50. Scaling values are calculated using training data only. Forward selection uses training-set Wald p-values with alpha = 0.05. The L2 value is chosen from a validation part of the training data instead of the test set.
