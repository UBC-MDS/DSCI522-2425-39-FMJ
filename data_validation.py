import click
import pandas as pd
import pandera as pa
import json
import logging

# Configure logging
logging.basicConfig(
    filename="validation_errors.log",
    filemode='w',
    format="%(asctime)s - %(message)s",
    level=logging.INFO
)

@click.command()
@click.option('--raw_data_path', type=str)
@click.option('--write_to', type=str)

def main(raw_data_path, write_to):

    original_df = pd.read_csv(raw_data_path)

    # Validate data and handle errors
    schema = pa.DataFrameSchema(
        {
            "SEQN": pa.Column(float, pa.Check.ge(0)), # ID is greater than 0
            "age_group": pa.Column(str, pa.Check.isin(["Adult", "Senior"])),
            "RIDAGEYR": pa.Column(float, pa.Check.between(1, 130)), # Age between 1 and 130
            "RIAGENDR": pa.Column(float, pa.Check.isin([1, 2])), # Gender follows codes in description doc
            "PAQ605": pa.Column(float, pa.Check.isin([1, 2])), # Physical activity follows codes in description doc
            "BMXBMI": pa.Column(float, pa.Check.le(100)), #BMI
            "LBXGLU": pa.Column(float), # Units not available, no value range applied
            "DIQ010": pa.Column(float, pa.Check.isin([1, 2])), # Diabetic status follows codes in description doc
            "LBXGLT": pa.Column(float), # Units not available, no value range applied
            "LBXIN": pa.Column(float) # Units not available, no value range applied
        },
        checks=[
            pa.Check(lambda df: ~df.duplicated().any(), error="Duplicate rows found."),
            pa.Check(lambda df: ~(df.isna().all(axis=1)).any(), error="Empty rows found.")
        ],
        strict=True #Raise error for any columns not defined in the schema
    )

    error_cases = pd.DataFrame()

    try:
        validated_data = schema.validate(original_df, lazy=True)
    except pa.errors.SchemaErrors as e:
        error_cases = e.failure_cases

        error_message = json.dumps(e.message, indent=2)
        logging.error("\n" + error_message)

    if not error_cases.empty:
        invalid_indices = error_cases['index'].dropna().unique()
        validated_data = (original_df.drop(index=invalid_indices).reset_index(drop=True)
        )

    validated_data = validated_data.drop_duplicates().dropna(how='all')

    validated_data.to_csv(write_to, index=False)

if __name__ == '__main__':
    main()