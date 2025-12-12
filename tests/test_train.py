import os
import pandas as pd
from sklearn.linear_model import LogisticRegression

DATA_PATH = "data/dataset.csv"

def test_dataset_exists():
    assert os.path.exists(DATA_PATH), f"{DATA_PATH} not found"

def test_data_load_and_shape():
    df = pd.read_csv(DATA_PATH)
    assert df.shape[0] > 0, "Dataset has no rows"
    assert df.shape[1] >= 2, "Expect at least one feature + target"

def test_model_training_and_predict():
    df = pd.read_csv(DATA_PATH)
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    y_enc = pd.factorize(y)[0]

    model = LogisticRegression(max_iter=200)
    model.fit(X, y_enc)

    preds = model.predict(X)
    assert preds.shape[0] == X.shape[0]
