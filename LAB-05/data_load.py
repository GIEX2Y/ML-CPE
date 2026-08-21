import pandas as pd


def load_data(file_path):
    """
    Load Breast Cancer dataset
    """

    df = pd.read_csv(file_path)

    # Remove unnecessary columns
    if "id" in df.columns:
        df = df.drop(columns=["id"])

    if "Unnamed: 32" in df.columns:
        df = df.drop(columns=["Unnamed: 32"])

    # Convert diagnosis
    df["diagnosis"] = df["diagnosis"].map({
        "B": 0,
        "M": 1
    })

    X = df.drop(columns=["diagnosis"])
    y = df["diagnosis"]

    return X, y