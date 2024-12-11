# validate_tran_data.py
# author: Forgive Agbesi
# date: 2024-12-10



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
