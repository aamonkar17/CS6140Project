# Instructions on how to run the project:

## Installation
pip install numpy pandas scikit-learn lightgbm xgboost optuna matplotlib seaborn scipy joblib tensorflow

## How to Run (in order)

Step 1 — Preprocessing:
python preprocessing.py

Step 2 — Feature Engineering:
python feature_engineering.py

Step 3 — Baseline Models:
python baseline_models.py

Step 4 — Advanced Models (Regression):
python advanced_models_regression.py

Step 4 — Advanced Models (Classification):
python advanced_models_classification.py

NOTE: Run Step 4 before Step 5 — Step 5 loads the saved XGBoost model from Step 4.

## Outputs
All outputs are saved to the "outputs/" folder:
- outputs/predictions/  — Results
- outputs/models/       — saved model files
- outputs/figures/      — plots and visualizations

## Datsets
Train.csv: https://drive.google.com/file/d/1PinXgLUh9SmL9ATDR7N2AWc8HGz-d9vK/view?usp=drive_link

Test.csv: https://drive.google.com/file/d/16se6Rg0M9_f6OMvKqupUv7_eBSBzWMYj/view?usp=drive_link

## Video
Video: 

## Github
Github link: https://github.com/aamonkar17/CS6140Project