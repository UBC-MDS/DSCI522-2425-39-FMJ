# fit_age_group_classifier.py
# author: Forgive Agbesi
# date: 2024-12-3

import click
import os
import altair as alt
import numpy as np
import pandas as pd
import pickle
from sklearn.dummy import DummyClassifier  
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler 
from sklearn.model_selection import cross_validate
from deepchecks.tabular import Dataset
from sklearn import set_config
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="deepchecks")


@click.command()
@click.option('--x_train_path', type=str, help="filepath of X_train.csv")
@click.option('--y_train_path', type=str, help="filepath of y_train.csv")
@click.option('--write_to', type=str, help="input path/<filename>.png")
@click.option('--pipeline-to', type=str, help="Path to directory where the pipeline object will be written to")
@click.option('--results-to', type=str, help="Path to directory where the csv will be written to")
@click.option('--seed', type=int, help="Random seed", default=123)

def mean_cross_val_scores(model, X_train, y_train, **kwargs):
    """
    Returns mean and std of cross validation

    Parameters  
    ----------
        model :
            scikit-learn model
        X_train : numpy array or pandas DataFrame
            X in the training data
        y_train :
            y in the training data
    Returns
    ----------
        pandas Series with mean scores from cross_validation
    """
    scores = cross_validate(model, X_train, y_train, **kwargs)
    mean_scores = pd.DataFrame(scores).mean()
    out_col = []

    for i in range(len(mean_scores)):
        out_col.append((mean_scores.iloc[i]))
    return pd.Series(data=out_col, index=mean_scores.index)


def main(X_training_data, y_training_data, preprocessor, pipeline_to, results_to, seed):
    '''Fits the age group classifier to the training data 
    and saves the pipeline object.'''

    np.random.seed(seed)
    set_config(transform_output="pandas")

    # read in data & preprocessor
    X_train = pd.read_csv(X_training_data)
    y_train = pd.read_csv(y_training_data)

    # Transforming columns based on their type
    numeric_features = ["bmi", "blood_glucose", "oral", "blood_insulin"]
    binary_features = ["gender", "physical_activity", "diabetic"]

    preprocessor = make_column_transformer(
        (OneHotEncoder(sparse_output = False,
                   drop='if_binary',dtype = int), binary_features),
        (StandardScaler(), numeric_features)
    )
   

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

    # save model_cv_score as image or file to be read in report
    model_cv_score.to_csv(os.path.join(results_to, "model_cv_score.csv"), index=False)
    
    # fitting best model 
    lr_pipe_fit = lr_pipe.fit(X_train, y_train)

    with open(os.path.join(pipeline_to, "LogisticRegression_classifier_pipeline.pickle"), 'wb') as f:
        pickle.dump(lr_pipe_fit, f)

if __name__ == '__main__':
    main()