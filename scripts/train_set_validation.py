# train_set_validation.py
# author: Forgive Agbesi
# date: 2024-12-3

import click
import pandas as pd
from deepchecks.tabular.checks import FeatureLabelCorrelation, FeatureFeatureCorrelation
from deepchecks.tabular import Dataset
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="deepchecks")


@click.command()
@click.option('--X_training_data', type=str, help="Path to X training data")
@click.option('--y_training_data', type=str, help="Path to y training data")
@click.option('--preprocessor', type=str, help="Path to preprocessor object")
@click.option('--pipeline-to', type=str, help="Path to directory where the pipeline object will be written to")
@click.option('--plot-to', type=str, help="Path to directory where the plot will be written to")
@click.option('--seed', type=int, help="Random seed", default=123)

# Validation check: Target/response variable follows expected distribution
def validate_category_distribution(y_train, age_group_thresholds, tolerance):
    """
    Validate if a categorical variable's distribution meets specified thresholds with tolerance.

    Parameters:
    - y_train (pd.Series): The categorical variable (target/response variable).
    - age_group_thresholds (dict): Minimum and maximum proportion thresholds for each category.
    - tolerance (float): The tolerance to apply when checking proportions.

    Returns:
    - bool: True if the distribution meets the thresholds with tolerance, False otherwise.
    """
    value_counts = y_train.value_counts(normalize=True)  # Get proportions

    # Loop through each category and its thresholds
    for category, (min_threshold, max_threshold) in age_group_thresholds.items():
        proportion = value_counts.get(category, 0)  # Get proportion for the category
        
        # Check if the proportion is within the threshold range with tolerance
        if not (min_threshold - tolerance <= proportion <= max_threshold + tolerance):
            return False  # Return False if the proportion is out of the acceptable range
    
    return True  # Return True if all categories meet the criteria

def main(y_training_data,X_training_data):
    age_group_thresholds = {"Adult": (0.2, 0.9), "Senior": (0.2, 0.9)}
    tolerance = 0.05

    # Validate the distribution
    is_valid = validate_category_distribution(y_training_data, age_group_thresholds, tolerance)
    print(is_valid)


    # validate training data for anomalous correlations between target/response variable 
    # and features/explanatory variables, 
    # as well as anomalous correlations between features/explanatory variables
    train_df = pd.concat([X_training_data, y_training_data], axis = 1)

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
   
if __name__ == '__main__':
    main()