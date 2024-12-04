# download_data.py
# author: Jason Lee
# date: 2024-12-03

# This script uses the ucirepo package to download the raw data, store it in a dataframe, and save it in the /data directory

# Usage:
# python scripts/download_data.py \
#    --repo_id=887 \
#    --write_to=data/Raw/NHANES_age_prediction.csv

import click
from ucimlrepo import fetch_ucirepo 

@click.command()
@click.option('--repo_id', type=int, help="id number of the uci repo")
@click.option('--write_to', type=str, help="input path/<filename>.csv")

def main(repo_id, write_to):
    """
    Uses the ucirepo package to download the raw data, store it in a dataframe, and save it in the /data directory
    """
    nhanes = fetch_ucirepo(id=repo_id)
    raw_data = nhanes.data.original
    raw_data.to_csv(write_to)

if __name__ == '__main__':
    main()