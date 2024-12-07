# Age Group Prediction 
## Can the health and nutritional status of Americans be used to classify their age group?

## Authors:
- Forgive Agbesi
- Jason Lee
- Michael Hewlett

## Summary:
Using a SVM model, we try to classify the age group ("senior", "non-senior") of a given person based on a few of their of health indicators.

## Report
The final report can be found [here](https://github.com/UBC-MDS/DSCI522-2425-39-FMJ/blob/main/notebooks/age_group_classification.ipynb)

## Dependencies
- [Docker](https://www.docker.com)

## Usage

### Setup

>If you are using Windows or Mac, make sure Docker Desktop is running.

1. Clone this GitHub Repository

```bash
git clone https://github.com/UBC-MDS/DSCI522-2425-39-FMJ
```

### Running the analysis

1. Navigate to the root of this project on your computer using the command line and enter the following command:

```bash
docker compose up
```

2. In the terminal, look for a URL that starts with http://127.0.0.1:8888/lab?token= (for an example, see the highlighted text in the terminal below). Copy and paste that URL into your browser.

3. To run the analysis, open a terminal and run the following commands:

```bash
python scripts/download_data.py \
   --repo_id=887 \
   --write_to=data/Raw/NHANES_age_prediction.csv

python scripts/validate_data.py \
   --raw_data_path=data/Raw/NHANES_age_prediction.csv \
   --write_to=data/processed/validated_data.csv

python scripts/data_split.py \
   --validated_data_path=data/processed/validated_data.csv \
   --write_to=data/processed

python scripts/eda.py \
    --x_train_path=data/processed/X_train.csv \
    --y_train_path=data/processed/y_train.csv \
    --write_to=reports/figures/eda_histogram.png

```

3. To run the analysis, open src/age_group_classification.ipynb in Jupyter Lab you just launched and under the "Kernel" menu click "Restart Kernel and Run All Cells..."

### Clean up

1. To shut down the container and clean up the resources, type `Cntrl` + `C` in the terminal where you launched the container, and then type `docker compose rm`

## Developer notes

### Developer dependencies
- `conda` (version 23.9.0 or higher)
- `conda-lock` (version 2.5.7 or higher)

### Adding a new dependency
1. Add the dependency to the environment.yml file on a new branch.

2. Run conda-lock -k explicit --file environment.yml -p linux-64 to update the conda-linux-64.lock file.

3. Re-build the Docker image locally to ensure it builds and runs properly.

4. Push the changes to GitHub. A new Docker image will be built and pushed to Docker Hub automatically. It will be tagged with the SHA for the commit that changed the file.

5. Update the docker-compose.yml file on your branch to use the new container image (make sure to update the tag specifically).

6. Send a pull request to merge the changes into the main branch.

## License
This project is licensed under a MIT License. See the license file for more information.

## References
Dua, Dheeru, and Casey Graff. 2017. “UCI Machine Learning Repository.” University of California, Irvine, School of Information; Computer Sciences. http://archive.ics.uci.edu/ml.

NA, N. (2019). National Health and Nutrition Health Survey 2013-2014 (NHANES) Age Prediction Subset [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5BS66.

Mukhtar, Hamid and Sana Al Azwari. “Investigating Non-Laboratory Variables to Predict Diabetic and Prediabetic Patients from Electronic Medical Records Using Machine Learning.” (2021).

Papazafiropoulou, Athanasia K.. “Diabetes management in the era of artificial intelligence.” Archives of Medical Sciences. Atherosclerotic Diseases 9 (2024): e122 - e128.
