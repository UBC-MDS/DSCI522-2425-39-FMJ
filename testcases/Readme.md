# Test suite developer notes

This repository contains test cases for validating different functionalities in the project. The test cases are written in Python and utilize the pytest framework for running and validating the results.

### Prerequisites

Before running the test cases, ensure the following dependencies are installed:

1. pytest (pip install pytest) 

### Running the tests

Tests are executed by running the 'pytest' command from the root directory of the project.

### Test Files Overview

The following test files are included:

1. test_mean_cross_validation_score.py
Tests the mean_cross_val_scores function.
Validates classification model performance (e.g., RandomForestClassifier, LogisticRegression) using cross-validation.

2. test_validate_train_data.py
Tests the validate_category_distribution function.
Ensures that categorical data distributions meet specified thresholds with tolerance.
Covers cases such as valid distribution, invalid distribution, missing categories, empty input, and wider tolerance.

3. test_write_csv.py
Tests CSV file writing functionality.
Validates if data is correctly written to CSV files.
Running Tests


