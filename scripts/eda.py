# eda.py
# author: Jason Lee
# date: 2024-12-03

# This script will take in X_train and y_train to create a histogram using Altair for EDA purposes.
# The histogram is saved as a png to a specified save path

# Usage:
# python scripts/eda.py \
#    --x_train_path=data/processed/X_train.csv \
#    --y_train_path=data/processed/y_train.csv \
#    --write_to=reports/figures/eda_histogram.png

import click
import altair as alt
import pandas as pd

@click.command()
@click.option('--x_train_path', type=str, help="filepath of X_train.csv")
@click.option('--y_train_path', type=str, help="filepath of y_train.csv")
@click.option('--write_to', type=str, help="input path/<filename>.png")

def main(x_train_path, y_train_path, write_to):
    """
    This script will take in the X_train and y_train and create a histogram using Altair for EDA purposes.
    """

    # Plotting the distributions for each variable
    alt.renderers.enable('png')

    X_train = pd.read_csv(x_train_path)
    y_train = pd.read_csv(y_train_path)
    features_list = X_train.columns.tolist()

    eda_histogram = alt.Chart(pd.concat([X_train, y_train], axis = 1)).mark_bar(opacity = 1).encode(
                x=alt.X(alt.repeat()).type('quantitative').bin(maxbins=40).stack(False),
                y='count()',
                color = 'age_group'
            ).repeat(
                features_list,
                columns = 2
            )

    eda_histogram.save(write_to)

if __name__ == '__main__':
    main()