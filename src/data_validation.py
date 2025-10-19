import pandas as pd

def validate_data(path="data/iris.csv"):
    df = pd.read_csv(path)
    assert not df.isnull().values.any(), "Data contains nulls!"
    assert df.shape[1] == 5, "Unexpected number of columns!"
    return True

if __name__ == "__main__":
    validate_data()
