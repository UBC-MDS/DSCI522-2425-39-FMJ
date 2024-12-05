# evaluate_age_group_classifier.py
# author: Forgive Agbesi
# date: 2024-12-3

import click
import os
import numpy as np
import pandas as pd
import pickle
from sklearn import set_config
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay


@click.command()
@click.option('--X-test-data', type=str, help="Path to scaled test data")
@click.option('--y-test-data', type=str, help="Optional: columns to drop")
@click.option('--pipeline-from', type=str, help="Path to directory where the fit pipeline object lives")
@click.option('--results-to', type=str, help="Path to directory where the plot will be written to")
@click.option('--seed', type=int, help="Random seed", default=123)

def main(X_test_data, y_test_data, pipeline_from, results_to, seed):
    '''Evaluates the age group classifier on the test data 
    and saves the evaluation results.'''
    np.random.seed(seed)
    set_config(transform_output="pandas")

    # read in data & cancer_fit (pipeline object)
    X_test = pd.read_csv(X_test_data)
    y_test = pd.read_csv(y_test_data)

    with open(pipeline_from, 'rb') as f:
        best_model = pickle.load(f)

    # Compute accuracy
    accuracy = best_model.score(X_test,y_test)

 

    test_confMatrix = ConfusionMatrixDisplay.from_estimator(
        best_model,
        X_test,
        y_test,
        values_format="d",
    )
    
    # test_confMatrix
    # save image for report
    #
    #
    #
    test_RocCurve = RocCurveDisplay.from_estimator(
    best_model,
    X_test,
    y_test,
    pos_label= "Senior",
    )
    # test_confMatrix
    # save image for report
    #
    #
    #

   
    #confusion_matrix.to_csv(os.path.join(results_to, "confusion_matrix.csv"))

if __name__ == '__main__':
    main()