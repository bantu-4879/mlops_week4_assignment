from src.train import train_model
from src.evaluate import evaluate_model

def test_model_accuracy():
    train_model()
    acc = evaluate_model()
    assert acc > 0.8
