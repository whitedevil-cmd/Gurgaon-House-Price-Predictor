# Gurgaon-House-Price-Predictor

A Python-based machine learning project to predict house prices in Gurgaon using a Random Forest Regressor. The project handles preprocessing, model training, and inference in an automated pipeline.

## Features
- End-to-end data preprocessing with pipelines for numeric and categorical features.
- Stratified train-test split based on income category to ensure balanced data.
- Real-time training and inference workflow using joblib for saving/loading models.
- Predicts median house prices and exports results to CSV.

## Dataset
The project uses a housing dataset (CSV format) with the following key features:
- `median_income` (used for stratified sampling)
- `median_house_value` (target variable)
- `ocean_proximity` (categorical feature)
- Other numeric attributes representing housing characteristics

> Place your dataset CSV in the project folder and name it `housing.csv`.

## How It Works
1. Checks if a trained model exists (`model.pkl`).  
2. If not, trains the model using:
   - `RandomForestRegressor` for prediction.
   - `ColumnTransformer` pipeline for preprocessing (imputation, scaling, one-hot encoding).  
3. Saves the trained model and pipeline using `joblib`.  
4. If a trained model exists, performs inference on `input.csv` and saves predictions to `output.csv`.

## Usage
1. Clone the repository:
```bash
git clone https://github.com/your-username/gurgaon-house-price-predictor.git
cd gurgaon-house-price-predictor
