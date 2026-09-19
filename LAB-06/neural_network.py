import numpy as np

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score


def load_dataset():
    data = load_breast_cancer()

    X = data.data
    y = data.target

    return X, y, data


def prepare_data(X, y):
    # Split Train/Test
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    # Standardization
    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test


def train_model(X_train, y_train, hidden_layers=(50, 25), epochs=50):

    model = MLPClassifier(
        hidden_layer_sizes=hidden_layers,
        max_iter=epochs,
        random_state=42,
        learning_rate_init=0.001
    )

    model.fit(X_train, y_train)

    return model


def evaluate_model(model, X_train, X_test, y_train, y_test):

    train_prediction = model.predict(X_train)
    test_prediction = model.predict(X_test)

    train_accuracy = accuracy_score(
        y_train,
        train_prediction
    )

    test_accuracy = accuracy_score(
        y_test,
        test_prediction
    )

    return train_accuracy, test_accuracy


def compare_epochs(X_train, X_test, y_train, y_test):

    epochs_list = [10, 25, 50, 100]

    results = []

    for epochs in epochs_list:

        model = train_model(
            X_train,
            y_train,
            hidden_layers=(50, 25),
            epochs=epochs
        )

        train_acc, test_acc = evaluate_model(
            model,
            X_train,
            X_test,
            y_train,
            y_test
        )

        results.append({
            "Epochs": epochs,
            "Train Accuracy": train_acc * 100,
            "Test Accuracy": test_acc * 100
        })

    return results


def compare_configurations(X_train, X_test, y_train, y_test):

    configurations = {
        "10 neurons": (10,),
        "50 neurons": (50,),
        "50-25": (50, 25),
        "100-50": (100, 50)
    }

    results = []

    for name, layers in configurations.items():

        model = train_model(
            X_train,
            y_train,
            hidden_layers=layers,
            epochs=50
        )

        train_acc, test_acc = evaluate_model(
            model,
            X_train,
            X_test,
            y_train,
            y_test
        )

        results.append({
            "Configuration": name,
            "Train Accuracy": train_acc * 100,
            "Test Accuracy": test_acc * 100
        })

    return results