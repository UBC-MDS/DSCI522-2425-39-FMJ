# data_split.py
# author: Jason Lee
# date: 2024-12-03

# Takes in the validation data, renames the feature columns, and splits the data into train and test sets. 
# Returns X_train.csv, X_test.csv, y_train.csv, y_test.csv.

# Usage:
# python scripts/data_split.py \
#    --validated_data_path=data/processed/validated_data.csv \
#    --write_to=data/processed

import click
import pandas as pd
from sklearn.model_selection import train_test_split

@click.command()
@click.option('--validated_data_path', type=str, help="path of the validated_data.csv")
@click.option('--write_to', type=str, help="save_path")

def main(validated_data_path, write_to):
    """
    Takes in the validation data, renames the feature columns, and splits the data into train and test sets. 
    Returns X_train.csv, X_test.csv, y_train.csv, y_test.csv.
    """

    validated_data = pd.read_csv(validated_data_path)
    features = validated_data.drop(columns=["Unnamed: 0", "SEQN", "age_group", "RIDAGEYR"])

    #Renaming the features
    features.columns = ["gender", 
             "physical_activity", 
             "bmi", 
             "blood_glucose", 
             "diabetic", 
             "oral", 
             "blood_insulin"]
    
    #Defining the target
    target = validated_data["age_group"]

    # Splitting the data
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=123)

    X_train.to_csv(f"{write_to}/X_train.csv")
    X_test.to_csv(f"{write_to}/X_test.csv")
    y_train.to_csv(f"{write_to}/y_train.csv")
    y_test.to_csv(f"{write_to}/y_test.csv")

if __name__ == '__main__':
    main()