import joblib
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

def evaluate_model():
    model = joblib.load("model.joblib")
    X, y = load_iris(return_X_y=True)
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"Accuracy: {acc}")
    return acc

if __name__ == "__main__":
    evaluate_model()
