# evaluate_age_group_classifier.py
# author: Forgive Agbesi
# date: 2024-12-3

# This script will take in x_train and y_train data and best model's pipeline
# create a confusion matix and Receiver Operating Characteristic Curve
# Saves the image of both plots as a png to be read in final report


# Usage:
# python scripts/evaluate.py \
# --x_test_data=data/processed/X_test.csv \
# --y_test_data=data/processed/y_test.csv \
# --pipeline_from=results/models/LogisticRegression_classifier_pipeline.pickle \
# --matrix_results_to=results/figures/Confusion_matrix.png
# --roc_results_to=results/figures/ROC.png

import click
import os
import numpy as np
import pandas as pd
import pickle
import altair as alt
import plotly.graph_objects as go
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn import set_config


@click.command()
@click.option('--x_test_data', type=str, help="Path to scaled test data")
@click.option('--y_test_data', type=str, help="Optional: columns to drop")
@click.option('--pipeline_from', type=str, help="Path to directory where the fit pipeline object lives")
@click.option('--matrix_results_to', type=str, help="Path to directory where the images will be written to")
@click.option('--roc_results_to', type=str, help="Path to directory where the images will be written to")

def main(x_test_data, y_test_data, pipeline_from, matrix_results_to, roc_results_to):
    '''Evaluates the age group classifier on the test data 
    and saves the evaluation results.'''

    seed = 123
    np.random.seed(seed)
    set_config(transform_output="pandas")

    # read in data & cancer_fit (pipeline object)
    X_test = pd.read_csv(x_test_data)
    y_test = pd.read_csv(y_test_data)

    with open(pipeline_from, 'rb') as f:
        best_model = pickle.load(f)

    # Compute accuracy
    accuracy = best_model.score(X_test,y_test)
    # test_confMatrix
    cm = confusion_matrix(y_test, best_model.predict(X_test))
    labels = ["Adult", "Seniors"]

    # Plotting the distributions for each variable
    alt.renderers.enable('png')
        # Convert the confusion matrix into a DataFrame for Altair
    cm_df = pd.DataFrame(cm, columns=labels, index=labels).reset_index()
    cm_df = cm_df.melt(id_vars="index", var_name="Predicted", value_name="Count")
    cm_df.rename(columns={"index": "Actual"}, inplace=True)

    # Create a heatmap
    conf_matrix_chart = alt.Chart(cm_df).mark_rect().encode(
        x=alt.X("Predicted:N", title="Predicted"),
        y=alt.Y("Actual:N", title="Actual"),
        color=alt.Color("Count:Q", scale=alt.Scale(scheme="blues"), title="Count"),
        tooltip=["Actual", "Predicted", "Count"]
    ).properties(
        title="Confusion Matrix",
        width=300,
        height=300
    )

    # Add text annotations for counts
    text = conf_matrix_chart.mark_text(baseline="middle").encode(
        text="Count:Q",
        color=alt.value("black")
    )

    # Combine the heatmap and text
    conf_matrix_with_text = conf_matrix_chart + text
    
    # Save the chart
    conf_matrix_with_text.save(matrix_results_to)
    
    # ROC Chart for model
    y_test_proba = best_model.predict_proba(X_test)[:, 1] 
    fpr, tpr, thresholds = roc_curve(y_test, y_test_proba, pos_label='Senior')
    roc_auc = auc(fpr, tpr)

    # Prepare the DataFrame for Altair
    roc_data = pd.DataFrame({
        "False Positive Rate": fpr,
        "True Positive Rate": tpr,
        "Thresholds": thresholds
    })

    # Create the ROC curve chart with Altair
    roc_chart = (
        alt.Chart(roc_data)
        .mark_line()
        .encode(
            x=alt.X("False Positive Rate", title="False Positive Rate"),
            y=alt.Y("True Positive Rate", title="True Positive Rate"),
            tooltip=["Thresholds"]
        )
        .properties(
            title=f"ROC Curve (AUC = {roc_auc:.2f})",
            width=600,
            height=400
        )
    )

    # Add a diagonal line for reference (random guessing)
    diagonal = (
        alt.Chart(pd.DataFrame({"x": [0, 1], "y": [0, 1]}))
        .mark_line(strokeDash=[5, 5], color="gray")
        .encode(
            x="x",
            y="y"
        )
    )

    # Combine the ROC curve and diagonal
    final_chart = roc_chart + diagonal

    # Save the chart
    final_chart.save(roc_results_to)

if __name__ == '__main__':
    main()