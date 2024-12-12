# Makefile
# Jason Lee, Dec 2024

# This driver script completes the age classification analysis using nutritional data. 
# Data is loaded, validated, preprocessed, then split. EDA is conducted in which a histogram is created.
# The model is then fitted and evaluated. Then a final report is created and exported as a pdf.
# This script takes no arguments.

# example usage:
# make all

.PHONY: all dats eda fit-model eval-model clean-dats clean-fit-model clean-eval-model clean-all

all: reports/age_group_classification.pdf

# Load Data / Data validation / Data splitting
dats: data/Raw/NHANES_age_prediction.csv \
data/processed/validated_data.csv \
data/processed/X_test.csv \
data/processed/X_train.csv \
data/processed/y_test.csv \
data/processed/y_train.csv

data/Raw/NHANES_age_prediction.csv : scripts/download_data.py
	python scripts/download_data.py \
		--repo_id=887 \
		--write_to=data/Raw/NHANES_age_prediction.csv

data/processed/validated_data.csv : data/Raw/NHANES_age_prediction.csv scripts/validate_data.py
	python scripts/validate_data.py \
		--raw_data_path=data/Raw/NHANES_age_prediction.csv \
		--write_to=data/processed/validated_data.csv

data/processed/X_test.csv \
data/processed/X_train.csv \
data/processed/y_test.csv \
data/processed/y_train.csv : data/processed/validated_data.csv scripts/data_split.py
	python scripts/data_split.py \
		--validated_data_path=data/processed/validated_data.csv \
		--write_to=data/processed

# EDA
eda: results/figures/eda_histogram.png

results/figures/eda_histogram.png : data/processed/X_train.csv data/processed/y_train.csv scripts/eda.py
	python scripts/eda.py \
		--x_train_path=data/processed/X_train.csv \
		--y_train_path=data/processed/y_train.csv \
		--write_to=results/figures/eda_histogram.png

# Fit model
fit-model: results/models/LogisticRegression_classifier_pipeline.pickle \
results/models/PreprocessorPipeline.pickle \
results/tables/model_cv_score.csv

results/models/LogisticRegression_classifier_pipeline.pickle \
results/models/PreprocessorPipeline.pickle \
results/tables/model_cv_score.csv : data/processed/X_train.csv data/processed/y_train.csv scripts/fit_classifier.py
	python scripts/fit_classifier.py \
		--x_training_data=data/processed/X_train.csv \
		--y_training_data=data/processed/y_train.csv \
		--pipeline_to=results/models \
		--preprocessor_to=results/models \
		--results_to=results/tables

# Evaluate model
eval-model: results/figures/Confusion_matrix.png \
results/figures/ROC.png

results/figures/Confusion_matrix.png \
results/figures/ROC.png : data/processed/X_test.csv data/processed/y_test.csv results/models/LogisticRegression_classifier_pipeline.pickle scripts/evaluate.py
	python scripts/evaluate.py \
		--x_test_data=data/processed/X_test.csv \
		--y_test_data=data/processed/y_test.csv \
		--pipeline_from=results/models/LogisticRegression_classifier_pipeline.pickle \
		--matrix_results_to=results/figures/Confusion_matrix.png \
		--roc_results_to=results/figures/ROC.png

# Write the report
reports/age_group_classification.pdf : reports/age_group_classification.qmd dats eda fit-model eval-model 
	quarto render reports/age_group_classification.qmd

clean-dats:
	rm -f data/Raw/NHANES_age_prediction.csv \
		data/processed/validated_data.csv \
		data/processed/X_test.csv \
		data/processed/X_train.csv \
		data/processed/y_test.csv \
		data/processed/y_train.csv 

clean-eda:
	rm -f results/figures/eda_histogram.png

clean-fit-model:
	rm -f results/models/LogisticRegression_classifier_pipeline.pickle \
		results/models/PreprocessorPipeline.pickle \
		results/tables/model_cv_score.csv

clean-eval-model:
	rm -f results/figures/Confusion_matrix.png \
		results/figures/ROC.png

clean-all: clean-dats \
	clean-eda \
	clean-fit-model \
	clean-eval-model
	rm -f reports/age_group_classification.pdf