from sklearn.svm import SVC


def train_svm(
        X_train,
        y_train,
        kernel="linear"
):
    """
    Train SVM model
    """

    model = SVC(
        kernel=kernel,
        random_state=42
    )

    model.fit(X_train, y_train)

    return model


def predict_svm(
        model,
        X_test
):
    """
    Predict labels
    """

    return model.predict(X_test)