# Age Group Prediction 
## Can the health and nutritional status of Americans be used to classify their age group?

## Authors:
- Forgive Agbesi
- Jason Lee
- Michael Hewlett

## Summary:
In this analysis we explored the use of several classification models to predict whether a respondent is an adult or senior (essentially below or above age 65) based on their health and nutritional data. Our most promising model used Logistic Regression. While it appeared promising, much of the model’s accuracy was achieved by classifying most respondents as adults, since this was the majority class. Precision and recall for predicting the senior class was quite low. This suggests that the model has considerable room for improvement, which could be achieved through optimizing the hyperparameters and selecting models based on precision, recall, or f1 scores, rather than general accuracy. With the goal of correctly classifying each group, false positive and false negative errors were both equally important for our analysis, and applying class weighting is worth exploring in future research. Once the model performs better on those metrics, it would be worth exploring which health and nutritional features are most predictive of age, which could provide suggestions for strategic public health programs.

## Report
The final report can be found [here](https://github.com/UBC-MDS/DSCI522-2425-39-FMJ/blob/main/reports/age_group_classification.pdf)

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

2. In the terminal, look for a URL that starts with http://127.0.0.1:8888/lab?token= . Copy and paste that URL into your browser.

3. Navigate to the root of this project on your computer using the command line and enter the following command to reset the project to a clean state (i.e., remove all files generated by previous runs of the analysis):

```bash
make clean-all
```

4. To run the analysis in its entirety, enter the following command in the terminal in the project root:
   
```bash
make all
```

5. To render the final report, enter the following command in the terminal in the project root:

```bash
quarto render reports/age_group_classification.qmd --to pdf
```

### Clean up

1. To shut down the container and clean up the resources, type `Ctrl` + `C` in the terminal where you launched the container, and then type `docker compose rm`

## Developer notes

### Developer dependencies
- `conda` (version 23.9.0 or higher)
- `conda-lock` (version 2.5.7 or higher)

### Adding a new dependency
1. Add the dependency to the `environment.yml` file on a new branch.

2. Run `conda-lock -k explicit --file environment.yml -p linux-64` to update the conda-linux-64.lock file.

3. Re-build the Docker image locally to ensure it builds and runs properly.

4. Push the changes to GitHub. A new Docker image will be built and pushed to Docker Hub automatically. It will be tagged with the SHA for the commit that changed the file.

5. Update the `docker-compose.yml` file on your branch to use the new container image (make sure to update the tag specifically).

6. Send a pull request to merge the changes into the main branch.

## License
This project is licensed under a MIT License. See the license file for more information.

## References
Bantilan, N. (2020). Pandera: Statistical Data Validation of Pandas Dataframes. In M. Agarwal, C. Calloway, D. Niederhut, & D. Shupe (Eds.), Proceedings of the 19th Python in Science Conference (pp. 116-124). https://doi.org/10.25080/Majora-342d178e-010

Chorev, S., Tannor, P., Ben Israel, D., Bressler, N., Gabbay, I., Hutnik, N., Liberman, J., Perlmutter, M., Romanyshyn, Y., & Rokach, L. (2022). Deepchecks: A Library for Testing and Validating Machine Learning Models and Data. Journal of Machine Learning Research, 23, 1–6. http://jmlr.org/papers/v23/22-0281.html

Dua, D., & Graff, C. (2017). UCI Machine Learning Repository. University of California, Irvine, School of Information and Computer Sciences. Retrieved from http://archive.ics.uci.edu/ml

Harris, C. R., Millman, K. J., van der Walt, S. J., Gommers, R., Virtanen, P., Cournapeau, D., Wieser, E., Taylor, J., Berg, S., Smith, N. J., Kern, R., Picus, M., Hoyer, S., van Kerkwijk, M. H., Brett, M., Haldane, A., Fernández del Río, J., Wiebe, M., Peterson, P., ... Oliphant, T. E. (2020). Array programming with NumPy. Nature, 585, 357-362. https://doi.org/10.1038/s41586-020-2649-2

NA, N. (2019). National Health and Nutrition Health Survey 2013-2014 (NHANES) Age Prediction Subset [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5BS66

Mukhtar, H., & Al Azwari, S. (2021). Investigating non-laboratory variables to predict diabetic and prediabetic patients from electronic medical records using machine learning.

Papazafiropoulou, A. K. (2024). Diabetes management in the era of artificial intelligence. Archives of Medical Sciences: Atherosclerotic Diseases, 9, e122–e128.

Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., Blondel, M., Prettenhofer, P., Weiss, R., Dubourg, V., Vanderplas, J., Passos, A., Cournapeau, D., Brucher, M., Perrot, M., & Duchesnay, É. (2011). Scikit-learn: Machine Learning in Python. Journal of Machine Learning Research, 12, 2825–2830. https://jmlr.csail.mit.edu/papers/v12/pedregosa11a.html

The pandas development team. pandas-dev/pandas: Pandas [Computer software]. https://doi.org/10.5281/zenodo.3509134

VanderPlas, J., Granger, B., Heer, J., Moritz, D., Wongsuphasawat, K., Satyanarayan, A., Lees, E., Timofeev, I., Welsh, B., & Sievert, S. (2018). Altair: Interactive Statistical Visualizations for Python. Journal of Open Source Software, 3(32), 1057. https://doi.org/10.21105/joss.01057

Van Rossum, G., & Drake, F. L. (2009). Python 3 Reference Manual. CreateSpace.
