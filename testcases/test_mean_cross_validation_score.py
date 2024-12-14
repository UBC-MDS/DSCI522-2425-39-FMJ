# test_mean_cross_validation_score.py
# author: Forgive Agbesi
# date: 2024-12-10

import pytest
import unittest
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.mean_cross_validation_score import mean_cross_val_scores


class test_mean_cross_val_scores(unittest.TestCase):

    def setUp(self):
        # Generate a simple classification dataset
        self.X, self.y = make_classification(
            n_samples=200, n_features=10, n_informative=5, n_redundant=0, random_state=42
        )

        # Split the dataset
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42
        )

        # Initialize models to test
        self.rf_model = RandomForestClassifier(random_state=42)
        self.lr_model = LogisticRegression(random_state=42, max_iter=500)

    def test_rf_model_cross_val_scores(self):
        # Test RandomForestClassifier
        rf_scores = mean_cross_val_scores(self.rf_model, self.X_train, self.y_train, cv=5, scoring="accuracy")
        
        self.assertIsInstance(rf_scores, pd.Series, "Output should be a pandas Series.")
        self.assertIn("test_score", rf_scores.index, "Expected 'test_score' in Series index.")
        self.assertGreater(rf_scores["test_score"], 0.5, "Mean accuracy should be above 0.5 for RandomForestClassifier.")

    def test_lr_model_cross_val_scores(self):
        # Test LogisticRegression
        lr_scores = mean_cross_val_scores(self.lr_model, self.X_train, self.y_train, cv=5, scoring="accuracy")
        
        self.assertIsInstance(lr_scores, pd.Series, "Output should be a pandas Series.")
        self.assertIn("test_score", lr_scores.index, "Expected 'test_score' in Series index.")
        self.assertGreater(lr_scores["test_score"], 0.5, "Mean accuracy should be above 0.5 for LogisticRegression.")
    
    def test_standard_deviation_inclusion(self):
        # Validate the standard deviation
        rf_scores = mean_cross_val_scores(self.rf_model, self.X_train, self.y_train, cv=5, scoring="accuracy")
        lr_scores = mean_cross_val_scores(self.lr_model, self.X_train, self.y_train, cv=5, scoring="accuracy")
        
        self.assertIn("test_score", lr_scores.index, "Expected 'test_score' in Series index.")
        self.assertIn("fit_time", rf_scores.index, "Expected 'fit_time' in Series index.")

    def test_finite_test_scores(self):
        # Check for consistent mean vs. computed values
        rf_scores = mean_cross_val_scores(self.rf_model, self.X_train, self.y_train, cv=5, scoring="accuracy")
        lr_scores = mean_cross_val_scores(self.lr_model, self.X_train, self.y_train, cv=5, scoring="accuracy")
        
        self.assertTrue(np.isfinite(rf_scores["test_score"]), "Mean test_score should be finite.")
        self.assertTrue(np.isfinite(lr_scores["test_score"]), "Mean test_score should be finite.")

if __name__ == "__main__":
    unittest.main()
