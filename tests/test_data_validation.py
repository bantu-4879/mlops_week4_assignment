from src.data_validation import validate_data

def test_data_validation():
    assert validate_data("data/iris.csv") == True
