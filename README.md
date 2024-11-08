# Project Title: Material-Structure-Processing-Performance Correlations for Polymeric Liner Material in Type-IV Hydrogen Pressure Vessels

## Overview
This project explores the use of machine learning to predict material permeability and permeation rates and to find the optimal material for hydrogen storage liners. It leverages a FastAPI backend for model predictions and a Streamlit frontend for user interaction.

## Table of Contents
- [Overview](#overview)
- [Project Structure](#project-structure)
- [Models Evaluated](#models-evaluated)
- [How to Run](#how-to-run)
- [Results](#results)

## Project Structure
- `app.py`: FastAPI backend for model predictions.
- `frontend.py`: Streamlit frontend for user input and visualization.
- `reports/`: Contains training graphs and profiling reports.
- `models/`: Pre-trained models and other supporting files.
- `requirements.txt`: List of required Python libraries.

## Models Evaluated

| Target  | Encoding      | Model             | CV R²      | MSE        | R²         |
| ------- | ------------- | ----------------- | ---------- | ---------- | ---------- |
| PER_FIT | One-Hot       | XGBoost           | -0.0001236 | 1.2082e-28 | -5.53e-06  |
| PER_FIT | One-Hot       | Random Forest     | -0.0001249 | 1.2082e-28 | -5.05e-06  |
| PER_FIT | One-Hot       | Gradient Boosting | -0.0001236 | 1.2082e-28 | -5.53e-06  |
| PER_FIT | One-Hot       | LightGBM          | 0.999665   | 3.8251e-32 | 0.999683   |
| PER_FIT | One-Hot       | CatBoost          | 0.999626   | 4.3415e-32 | 0.999641   |
| PER_FIT | Label         | XGBoost           | -0.0001236 | 1.2082e-28 | -5.53e-06  |
| PER_FIT | Label         | Random Forest     | -0.0001249 | 1.2082e-28 | -5.05e-06  |
| PER_FIT | Label         | Gradient Boosting | -0.0001236 | 1.2082e-28 | -5.53e-06  |
| PER_FIT | Label         | LightGBM          | 0.999667   | 3.8059e-32 | 0.999685   |
| PER_FIT | Label         | CatBoost          | 0.999625   | 4.3517e-32 | 0.999640   |
| PER_FIT | Target        | XGBoost           | -0.0001236 | 1.2082e-28 | -5.53e-06  |
| PER_FIT | Target        | Random Forest     | -0.0001249 | 1.2082e-28 | -5.05e-06  |
| PER_FIT | Target        | Gradient Boosting | -0.0001236 | 1.2082e-28 | -5.53e-06  |
| PER_FIT | Target        | LightGBM          | 0.999663   | 3.8362e-32 | 0.999682   |
| PER_FIT | Target        | CatBoost          | 0.999651   | 4.0719e-32 | 0.999663   |
| PER_FIT | Leave-One-Out | XGBoost           | -0.0001236 | 1.2082e-28 | -5.53e-06  |
| PER_FIT | Leave-One-Out | Random Forest     | -0.0001249 | 1.2082e-28 | -5.05e-06  |
| PER_FIT | Leave-One-Out | Gradient Boosting | -0.0001236 | 1.2082e-28 | -5.53e-06  |
| PER_FIT | Leave-One-Out | LightGBM          | 0.9999999  | 1.1175e-31 | 0.999075   |
| PER_FIT | Leave-One-Out | CatBoost          | 0.9999999  | 1.5802e-31 | 0.998692   |
| PR_NCC  | One-Hot       | XGBoost           | 0.962803   | 2.3095     | 0.962249   |
| PR_NCC  | One-Hot       | Random Forest     | 0.947523   | 3.6537     | 0.940277   |
| PR_NCC  | One-Hot       | Gradient Boosting | 0.960900   | 2.2343     | 0.963479   |
| PR_NCC  | One-Hot       | LightGBM          | 0.970527   | 1.8724     | 0.969394   |
| PR_NCC  | One-Hot       | CatBoost          | 0.966715   | 2.1249     | 0.965266   |
| PR_NCC  | Label         | XGBoost           | 0.963118   | 2.3413     | 0.961729   |
| PR_NCC  | Label         | Random Forest     | 0.947569   | 3.6510     | 0.940322   |
| PR_NCC  | Label         | Gradient Boosting | 0.941654   | 3.3553     | 0.945155   |
| PR_NCC  | Label         | LightGBM          | 0.970668   | 1.8488     | 0.969779   |
| PR_NCC  | Label         | CatBoost          | 0.966999   | 2.0931     | 0.965786   |
| PR_NCC  | Target        | XGBoost           | 0.963378   | 2.4278     | 0.960315   |
| PR_NCC  | Target        | Random Forest     | 0.947464   | 3.6552     | 0.940253   |
| PR_NCC  | Target        | Gradient Boosting | 0.968894   | 1.8664     | 0.969491   |
| PR_NCC  | Target        | LightGBM          | 0.970501   | 1.8948     | 0.969028   |
| PR_NCC  | Target        | CatBoost          | 0.966753   | 2.1285     | 0.965208   |
| PR_NCC  | Leave-One-Out | XGBoost           | 0.994827   | 20.7375    | 0.661026   |
| PR_NCC  | Leave-One-Out | Random Forest     | 0.999954   | 9.2029     | 0.849570   |
| PR_NCC  | Leave-One-Out | Gradient Boosting | 0.981889   | 3.0998     | 0.949332   |
| PR_NCC  | Leave-One-Out | LightGBM          | 0.996344   | 5.9428     | 0.902860   |
| PR_NCC  | Leave-One-Out | CatBoost          | 0.995746   | 14.2006    | 0.767878   |
| MC      | One-Hot       | XGBoost           | 0.999815   | 8241.68    | 0.999845   |
| MC      | One-Hot       | Random Forest     | 0.9999996  | 1.6044     | 0.99999997 |
| MC      | One-Hot       | Gradient Boosting | 0.995419   | 214866.31  | 0.995953   |
| MC      | One-Hot       | LightGBM          | 0.999809   | 10833.85   | 0.999796   |
| MC      | One-Hot       | CatBoost          | 0.999964   | 1707.12    | 0.999968   |
| MC      | Label         | XGBoost           | 0.999761   | 11447.33   | 0.999784   |
| MC      | Label         | Random Forest     | 0.9999996  | 1.1264     | 0.99999998 |
| MC      | Label         | Gradient Boosting | 0.992540   | 405738.00  | 0.992357   |
| MC      | Label         | LightGBM          | 0.999786   | 11789.74   | 0.999778   |
| MC      | Label         | CatBoost          | 0.999952   | 2277.94    | 0.999957   |
| MC      | Target        | XGBoost           | 0.999832   | 8818.99    | 0.999834   |
| MC      | Target        | Random Forest     | 0.9999996  | 1.5993     | 0.99999997 |
| MC      | Target        | Gradient Boosting | 0.996968   | 183451.00  | 0.996544   |
| MC      | Target        | LightGBM          | 0.999812   | 10120.23   | 0.999809   |
| MC      | Target        | CatBoost          | 0.999963   | 1762.26    | 0.999966   |
| MC      | Leave-One-Out | XGBoost           | 0.959835   | 13.1156e-5 | -1.553     |
| MC      | Leave-One-Out | Random Forest     | 0.966543   | 10.1117e-5 | -2.543     |
| MC      | Leave-One-Out | Gradient Boosting | 0.981678   | 9.2766e-5  | -1.866     |
| MC      | Leave-One-Out | LightGBM          | 0.999456   | 10.4378e-5 | -1.245     |
| MC      | Leave-One-Out | CatBoost          | 0.999564   | 6.1464e-5  | -0.643     |


## How to Run
### Backend (FastAPI)
1. Clone this repo and navigate to the project directory.
2. Install dependencies: `pip install -r requirements.txt`.
3. Run the backend using:
   ```bash
   uvicorn app:app --host 0.0.0.0 --port 8000
   ```

### Frontend (Streamlit)
1. Run the frontend using:
   ```bash
   streamlit run frontend.py
   ```
   
2. Update the API endpoint in `frontend.py` with the deployed FastAPI URL, if running in production.

## Results
Key findings and final model performance metrics:
- Best model: CatBoost (refer to the table for metrics).
- Graphs and profiling reports can be found in the `reports/` directory.
