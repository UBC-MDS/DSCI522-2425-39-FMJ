# fit_age_group_classifier.py
# author: Forgive Agbesi
# date: 2024-12-3

# This script will take in X_train and y_train data and perform crossvalidation on a DummyClassifier, a LogisticRegression and a SVC
# Saves the CV results table as a csv to be read in final report
# Will also save the Logistic Regression model as a pickle object

# Usage:
# python scripts/fit_classifier.py \
# --x_training_data=data/processed/X_train.csv \
# --y_training_data=data/processed/y_train.csv \
# --pipeline_to=results/models \
# --preprocessor_to=results/models \
# --results_to=results/tables

import click
import os
import numpy as np
import pandas as pd
import pickle
from deepchecks.tabular.checks import FeatureLabelCorrelation, FeatureFeatureCorrelation
from deepchecks.tabular import Dataset
from sklearn.dummy import DummyClassifier  
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler 
from sklearn import set_config
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.validate_train_data import validate_category_distribution
from src.mean_cross_validation_score import mean_cross_val_scores
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="deepchecks")

@click.command()
@click.option('--x_training_data', type=str, help="filepath of X_train.csv")
@click.option('--y_training_data', type=str, help="filepath of y_train.csv")
@click.option('--pipeline_to', type=str, help="Path to directory where the pipeline object will be written to")
@click.option('--preprocessor_to', type=str, help="Path to directory where the preprocessor will be written to")
@click.option('--results_to', type=str, help="Path to directory where the csv will be written to")
#@click.option('--seed', type=int, help="Random seed", default=123)
def main(x_training_data, y_training_data, pipeline_to, results_to, preprocessor_to):
    '''Fits the age group classifier to the training data 
    and saves the pipeline object.'''

    age_group_thresholds = {"Adult": (0.2, 0.9), "Senior": (0.2, 0.9)}
    tolerance = 0.05

    # read in data & preprocessor
    X_train = pd.read_csv(x_training_data)
    y_train = pd.read_csv(y_training_data)
    

    # Validate the distribution
    is_valid = validate_category_distribution(y_train, age_group_thresholds, tolerance)
    print(is_valid)


    # validate training data for anomalous correlations between target/response variable 
    # and features/explanatory variables, 
    # as well as anomalous correlations between features/explanatory variables
    train_df = pd.concat([X_train, y_train], axis = 1)

    # Specify categorical features if applicable
    categorical_features = ["gender", "physical_activity", "diabetic"]

    # Initialize Deepchecks Dataset
    train_df_ds = Dataset(train_df, label="age_group", cat_features=categorical_features)

    # Feature-Label Correlation Check
    check_feat_tar_corr = FeatureLabelCorrelation().add_condition_feature_pps_less_than(0.9)
    check_feat_tar_corr_result = check_feat_tar_corr.run(dataset=train_df_ds)

    # Feature-Feature Correlation Check
    check_feat_feat_cor = FeatureFeatureCorrelation().add_condition_max_number_of_pairs_above_threshold(threshold=0.92, n_pairs=0)
    check_feat_feat_cor_result = check_feat_feat_cor.run(dataset=train_df_ds)

    # Validate conditions
    if not check_feat_tar_corr_result.passed_conditions():
        raise ValueError("Feature-Label correlation exceeds the maximum acceptable threshold.")

    if not check_feat_feat_cor_result.passed_conditions():
        raise ValueError("Feature-Feature correlation exceeds the maximum acceptable threshold.")
   

    seed = 123
    np.random.seed(seed)
    set_config(transform_output="pandas")
    

    # Transforming columns based on their type
    numeric_features = ["bmi", "blood_glucose", "oral", "blood_insulin"]
    binary_features = ["gender", "physical_activity", "diabetic"]

    preprocessor = make_column_transformer(
        (OneHotEncoder(sparse_output = False,
                   drop='if_binary',dtype = int), binary_features),
        (StandardScaler(), numeric_features)
    )
    with open(os.path.join(preprocessor_to, "PreprocessorPipeline.pickle"), 'wb') as f:
        pickle.dump(preprocessor, f)
   
    y_train = y_train.values.ravel()
    # tune model (here, use default hyper parameter values)
    classifier_result = {}
    dummy = DummyClassifier(random_state = 123)
    dc_pipe = make_pipeline(preprocessor, dummy)
    classifier_result['Dummy']= mean_cross_val_scores(dc_pipe, X_train, y_train, cv=5, 
                                                        return_train_score=True)
                                                        
    lr = LogisticRegression(random_state = 123, class_weight='balanced')
    lr_pipe = make_pipeline(preprocessor, lr)
    classifier_result['Logistic']= mean_cross_val_scores(lr_pipe, X_train, y_train, cv=5, 
                                                         return_train_score=True)
                                                         
    svc = SVC(random_state = 123, class_weight='balanced')
    svc_pipe = make_pipeline(preprocessor, svc)
    classifier_result['SVC']= mean_cross_val_scores(svc_pipe, X_train, y_train, cv=5, 
                                                    return_train_score=True)
                                                         
    model_cv_score = pd.DataFrame(classifier_result).T
    model_cv_score = model_cv_score.drop(columns=['fit_time', 'score_time'])

    # save model_cv_score as csv to be read in report
    model_cv_score.to_csv(os.path.join(results_to, "model_cv_score.csv"))
    
    # fitting best model 
    lr_pipe_fit = lr_pipe.fit(X_train, y_train)

    with open(os.path.join(pipeline_to, "LogisticRegression_classifier_pipeline.pickle"), 'wb') as f:
        pickle.dump(lr_pipe_fit, f)

if __name__ == '__main__':
    main()

