CS 7830-01 Machine Learning
Assignment 2
SK Rakib Ul Islam Rahat

GitHub working project
----------------------
https://github.com/skrakibulislamrahat/CS7830-Assignment-2-Logistic-Regression

The GitHub repository contains the same code and dataset, a notebook for
running the project cell by cell, the report, generated results, and an
automatic run check.

Files included
--------------
Flu.csv                    dataset
CS7830_Assignment_2.py     Python code for Questions 1 through 4
Assignment_2_Report.pdf    report with results, graphs, and discussion
README.txt                 this file

The code uses Python 3, NumPy, pandas, and Matplotlib. I did not use
scikit-learn, statsmodels, or another machine-learning toolbox.

Install the three required packages if they are not already installed:

    python -m pip install numpy pandas matplotlib

Running the code
----------------
Keep Flu.csv and CS7830_Assignment_2.py in the same folder. Open a terminal
in that folder and run:

    python CS7830_Assignment_2.py

On a computer where the Python command is python3, run:

    python3 CS7830_Assignment_2.py

The script reads Flu.csv using a relative path. It creates a results folder
with the figures, CSV result tables, and summary.json. Existing result files
with the same names are replaced.

Reproducing my results
----------------------
I used random seed 42, a manually stratified 80/20 train-test split, and a
classification threshold of 0.50. Missing rows are removed only when a
variable needed for that part is missing. Scaling values are calculated from
the training data. Forward selection uses Wald p-values with alpha = 0.05,
and the L2 value is chosen with a validation part of the training data.
