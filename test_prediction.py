from src.pipeline.predict_pipeline import PredictPipeline

if __name__ == "__main__":
    
    sample_input = {
    "NAME_CONTRACT_TYPE": "Cash loans",
    "CODE_GENDER": "M",
    "DAYS_BIRTH": -12000,
    "AMT_INCOME_TOTAL": 150000,
    "AMT_CREDIT": 200000,
    "NUM_PREV_LOANS": 3,
    "HAS_PREV_LOANS": 1,
    "PREV_AMT_APPLICATION_MEAN": 180000,
    "PREV_AMT_CREDIT_MEAN": 170000
}


    # 👇 Give correct path to your artifacts
    preprocessor_path = "artifacts/preprocessor.pkl"
    model_path = "artifacts/model.pkl"

    # Initialize the pipeline
    pipeline = PredictPipeline(preprocessor_path, model_path)

    # Make prediction
    prediction = pipeline.predict(sample_input)

    print("🔥 Prediction:", prediction)
