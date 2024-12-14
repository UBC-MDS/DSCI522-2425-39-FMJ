# test_validate_train_data.py
# author: Forgive Agbesi
# date: 2024-12-11


import pytest
import pandas as pd
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.validate_train_data import validate_category_distribution

def test_validate_category_distribution():
    # Case 1: Valid category distribution
    y_train = pd.Series(["A", "A", "B", "B", "C", "C", "C", "A"])
    thresholds = {
        "A": (0.3, 0.5),
        "B": (0.2, 0.4),
        "C": (0.2, 0.5)
    }
    tolerance = 0.05
    assert validate_category_distribution(y_train, thresholds, tolerance) == True, "Valid distribution should pass."

    # Case 2: Invalid category distribution (A proportion too high)
    y_train = pd.Series(["A", "A", "A", "B", "B", "C", "C", "C"])
    thresholds = {
        "A": (0.1, 0.3),
        "B": (0.2, 0.4),
        "C": (0.2, 0.5)
    }
    assert validate_category_distribution(y_train, thresholds, tolerance) == False, "Distribution exceeding threshold should fail."

    # Case 3: Missing category
    thresholds = {
    "A": (0.1, 0.4),
    "B": (0.2, 0.4),
    "C": (0.2, 0.5),
    "D": (0.1, 0.3)  # Adding a missing category 'D'
}
    y_train = pd.Series(["A", "A", "B", "B", "B", "C", "C", "C"])
    tolerance = 0.05
    assert validate_category_distribution(y_train, thresholds, tolerance) == False, "Missing category should fail."

    # Case 4: All categories meet thresholds with tolerance
    y_train = pd.Series(["A", "A", "B", "B", "C", "C", "C", "A"])
    thresholds = {
        "A": (0.25, 0.5),
        "B": (0.2, 0.4),
        "C": (0.2, 0.5)
    }
    tolerance = 0.1
    assert validate_category_distribution(y_train, thresholds, tolerance) == True, "Valid distribution with wider tolerance should pass."

    # Case 5: Empty input
    y_train = pd.Series([])
    assert validate_category_distribution(y_train, thresholds, tolerance) == False, "Empty input should fail."
