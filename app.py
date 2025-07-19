from flask import Flask, request, render_template
import pandas as pd
import pickle
from src.utils import load_obj
from src.exception import CustomException
import sys
import os

app = Flask(__name__)

# Load model and preprocessor
preprocessor_path = "artifacts/preprocessor.pkl"
model_path = "artifacts/model.pkl"

preprocessor = load_obj(preprocessor_path)
model = load_obj(model_path)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = {
            'NAME_CONTRACT_TYPE': [request.form['NAME_CONTRACT_TYPE']],
            'CODE_GENDER': [request.form['CODE_GENDER']],
            'AMT_INCOME_TOTAL': [float(request.form['AMT_INCOME_TOTAL'])],
            'AMT_CREDIT': [float(request.form['AMT_CREDIT'])],
            'DAYS_BIRTH': [int(request.form['DAYS_BIRTH'])],
            'PREV_AMT_APPLICATION_MEAN': [float(request.form['PREV_AMT_APPLICATION_MEAN'])],
            'PREV_AMT_CREDIT_MEAN': [float(request.form['PREV_AMT_CREDIT_MEAN'])],
            'NUM_PREV_LOANS': [float(request.form['NUM_PREV_LOANS'])],
            'HAS_PREV_LOANS': [int(request.form['HAS_PREV_LOANS'])]
        }

        input_df = pd.DataFrame(data)

        # Preprocess the input
        input_scaled = preprocessor.transform(input_df)

        # Make prediction
        prediction = model.predict(input_scaled)

        # Format result
        result = f"Prediction: {'Loan Will Default' if prediction[0] == 1 else 'Loan Will Not Default'}"

        return render_template("index.html", prediction_text=result)

    except Exception as e:
        raise CustomException(e, sys)

if __name__ == "__main__":
    app.run(debug=True)
