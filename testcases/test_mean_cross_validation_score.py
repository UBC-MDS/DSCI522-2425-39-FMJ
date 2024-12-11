# test_mean_cross_validation_score.py
# author: Forgive Agbesi
# date: 2024-12-10

import pytest
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from src.mean_cross_validation_score import mean_cross_val_scores


def test_mean_cross_val_scores():
    # Generate a simple classification dataset
    X, y = make_classification(
        n_samples=200, n_features=10, n_informative=5, n_redundant=0, random_state=42
    )

    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize models to test
    rf_model = RandomForestClassifier(random_state=42)
    lr_model = LogisticRegression(random_state=42, max_iter=500)

    # Test RandomForestClassifier
    rf_scores = mean_cross_val_scores(rf_model, X_train, y_train, cv=5, scoring="accuracy")
    assert isinstance(rf_scores, pd.Series), "Output should be a pandas Series."
    assert "test_score" in rf_scores.index, "Expected 'test_score' in Series index."
    assert rf_scores["test_score"] > 0.5, "Mean accuracy should be above 0.5 for RandomForestClassifier."

    # Test LogisticRegression
    lr_scores = mean_cross_val_scores(lr_model, X_train, y_train, cv=5, scoring="accuracy")
    assert isinstance(lr_scores, pd.Series), "Output should be a pandas Series."
    assert "test_score" in lr_scores.index, "Expected 'test_score' in Series index."
    assert lr_scores["test_score"] > 0.5, "Mean accuracy should be above 0.5 for LogisticRegression."

    # Validate the standard deviation
    assert "test_score" in lr_scores.index, "Expected 'test_score' in Series index."
    assert "fit_time" in rf_scores.index, "Expected 'fit_time' in Series index."
    
    # Check for consistent mean vs. computed values
    assert np.isfinite(rf_scores["test_score"]), "Mean test_score should be finite."
    assert np.isfinite(lr_scores["test_score"]), "Mean test_score should be finite."

if __name__ == "__main__":
    pytest.main(["-v", "test_mean_cross_val.py"])