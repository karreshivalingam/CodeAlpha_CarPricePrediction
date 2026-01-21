import pandas as pd
import joblib

MODEL_PATH = "../models/random_forest_model.pkl"

def make_sample_input():
    """
    Creates a sample input row that matches the training features in X.csv.
    Update values to test different predictions.
    """
    # Load the processed feature columns to ensure correct order
    X_cols = pd.read_csv("../data/processed/X.csv").columns

    # Start with zeros for all features
    sample = pd.DataFrame([[0]*len(X_cols)], columns=X_cols)

    # Fill numeric features
    sample.loc[0, "Year"] = 2016
    sample.loc[0, "Present_Price"] = 7.5
    sample.loc[0, "Driven_kms"] = 45000
    sample.loc[0, "Owner"] = 0

    # One-hot encoded columns (may exist depending on your get_dummies output)
    # Set the correct category columns to 1.
    for col in sample.columns:
        if col == "Fuel_Type_Diesel":
            sample.loc[0, col] = 1   # Diesel
        if col == "Selling_type_Individual":
            sample.loc[0, col] = 0   # Dealer (0 means Dealer if Individual is the dummy)
        if col == "Transmission_Manual":
            sample.loc[0, col] = 1   # Manual

    return sample


def main():
    model = joblib.load(MODEL_PATH)
    sample = make_sample_input()
    pred = model.predict(sample)[0]
    print(f"Predicted Selling Price: {pred:.2f} lakhs")


if __name__ == "__main__":
    main()
