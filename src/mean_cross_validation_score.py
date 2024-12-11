# mean_cross_validation_score.py
# author: Forgive Agbesi
# date: 2024-12-10

import pandas as pd 
from sklearn.model_selection import cross_validate
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="deepchecks")

def mean_cross_val_scores(model, x_train, y_train, **kwargs):
    """
    Returns mean and std of cross validation

    Parameters  
    ----------
        model :
            scikit-learn model
        x_train : numpy array or pandas DataFrame
            X in the training data
        y_train :
            y in the training data
    Returns
    ----------
        pandas Series with mean scores from cross_validation
    """
    scores = cross_validate(model, x_train, y_train, **kwargs)
    mean_scores = pd.DataFrame(scores).mean()
    out_col = []

    for i in range(len(mean_scores)):
        out_col.append((mean_scores.iloc[i]))
    return pd.Series(data=out_col, index=mean_scores.index)

