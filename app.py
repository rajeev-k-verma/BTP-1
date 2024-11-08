from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI()

# Load models, scaler, and encoder
models = {target: joblib.load(f'model_{target}.pkl') for target in ['MC', 'PR_NCC', 'PER_FIT']}
scaler = joblib.load('scaler_X.pkl')
encoder = joblib.load('encoder_X.pkl')

class InputData(BaseModel):
    C: float
    LD: float
    DI: float
    TL: float
    SA_D: float
    SA_CYL: float
    PO: float
    M_NAME: str = None

class Predictions(BaseModel):
    MC: float
    PR_NCC: float
    PER_FIT: float
    M_NAME: str

@app.post("/predict_with_mname", response_model=Predictions)
def predict_with_mname(data: InputData):
    if not data.M_NAME:
        raise HTTPException(status_code=400, detail="M_NAME must be provided")

    input_data = data.dict()
    numerical_features = ['C', 'LD', 'DI', 'TL', 'SA_D', 'SA_CYL', 'PO']
    input_df = pd.DataFrame([input_data])
    input_df[numerical_features] = scaler.transform(input_df[numerical_features])
    input_df = encoder.transform(input_df)

    mc_pred = models['MC'].predict(input_df)[0]
    pr_ncc_pred = models['PR_NCC'].predict(input_df)[0]
    per_fit_pred = models['PER_FIT'].predict(input_df)[0]

    predictions = {
        'MC': mc_pred,
        'PR_NCC': pr_ncc_pred,
        'PER_FIT': per_fit_pred,
        'M_NAME': data.M_NAME
    }

    return predictions

@app.post("/find_optimal_mname", response_model=Predictions)
def find_optimal_mname(data: InputData):
    input_data = data.dict()
    numerical_features = ['C', 'LD', 'DI', 'TL', 'SA_D', 'SA_CYL', 'PO']
    materials = ['K-X09086', 'FLX40HP', 'HD614', 'B5500']

    optimal_material = None
    min_mc = float('inf')
    optimal_predictions = None

    for material in materials:
        input_data['M_NAME'] = material
        input_df = pd.DataFrame([input_data])
        input_df[numerical_features] = scaler.transform(input_df[numerical_features])
        input_df = encoder.transform(input_df)

        mc_pred = models['MC'].predict(input_df)[0]
        pr_ncc_pred = models['PR_NCC'].predict(input_df)[0]
        per_fit_pred = models['PER_FIT'].predict(input_df)[0]

        print(f"Evaluating material: {material}, MC: {mc_pred}, PR_NCC: {pr_ncc_pred}, PER_FIT: {per_fit_pred}")

        if pr_ncc_pred < 6:
            if mc_pred < min_mc:
                min_mc = mc_pred
                optimal_material = material
                optimal_predictions = {
                    'MC': mc_pred,
                    'PR_NCC': pr_ncc_pred,
                    'PER_FIT': per_fit_pred,
                    'M_NAME': material
                }

    if not optimal_predictions:
        raise HTTPException(status_code=404, detail="No optimal material found that meets the constraints.")

    predictions = optimal_predictions

    print(f"Optimal predictions: {predictions}")

    return predictions

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
