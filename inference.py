from src.pipeline.predict_pipeline import PredictPipeline

def run_inference(input_dict):
    preprocessor_path = "artifacts/preprocessor.pkl"
    model_path = "artifacts/model.pkl"

    pipeline = PredictPipeline(preprocessor_path, model_path)
    prediction = pipeline.predict(input_dict)
    return prediction

# Example usage
if __name__ == "__main__":
    input_data = {
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

    result = run_inference(input_data)
    print(" Final Prediction:", result)
