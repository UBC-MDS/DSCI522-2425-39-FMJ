# Age Group Prediction 
## Can the health and nutritional status of adults and children be used to classify age group?

## Authors:
- Forgive Agbesi
- Jason Lee
- Michael Hewlett

## Summary:
Using a SVM model, we try to classify the age group ("senior", "non-senior") of a given person based on a few of their of health indicators.

## Report
The final report can be found [here](https://github.com/UBC-MDS/DSCI522-2425-39-FMJ/blob/main/analysis.ipynb)

## Dependencies
This project requires specific dependencies, which are listed in the `environment.yml` and `conda-lock.yml` files. Follow these steps to set up the environment:

1. Download or Clone the Repository
```bash
git clone https://github.com/UBC-MDS/DSCI522-2425-39-FMJ
cd DSCI522-2425-39-FMJ
```
2. Create the Conda Environment using the environment.yml file

Option A: Using environment.yml
```bash
conda env create -f environment.yml
```

Option B: Using conda-lock.yml
```bash
conda-lock install --file conda-lock.yml
```

3. Activate the Environment

## License
This project is licensed under a MIT License. See the license file for more information.

## References
Dua, Dheeru, and Casey Graff. 2017. “UCI Machine Learning Repository.” University of California, Irvine, School of Information; Computer Sciences. http://archive.ics.uci.edu/ml.

NA, N. (2019). National Health and Nutrition Health Survey 2013-2014 (NHANES) Age Prediction Subset [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5BS66.