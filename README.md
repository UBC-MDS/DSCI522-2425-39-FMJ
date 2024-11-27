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

If you are using Windows or Mac, make sure Docker Desktop is running.

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

3. To run the analysis, open src/age_group_classification.ipynb in Jupyter Lab you just launched and under the "Kernel" menu click "Restart Kernel and Run All Cells..."

### Clean up

1. To shut down the container and clean up the resources, type `Cntrl` + `C` in the terminal where you launched the container, and then type `docker compose rm`

## License
This project is licensed under a MIT License. See the license file for more information.

## References
Dua, Dheeru, and Casey Graff. 2017. “UCI Machine Learning Repository.” University of California, Irvine, School of Information; Computer Sciences. http://archive.ics.uci.edu/ml.

NA, N. (2019). National Health and Nutrition Health Survey 2013-2014 (NHANES) Age Prediction Subset [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5BS66.

Mukhtar, Hamid and Sana Al Azwari. “Investigating Non-Laboratory Variables to Predict Diabetic and Prediabetic Patients from Electronic Medical Records Using Machine Learning.” (2021).

Papazafiropoulou, Athanasia K.. “Diabetes management in the era of artificial intelligence.” Archives of Medical Sciences. Atherosclerotic Diseases 9 (2024): e122 - e128.