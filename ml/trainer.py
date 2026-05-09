from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor


def train_baseline_model(X_train, y_train, task_type):

    if task_type == "classification":

        model = RandomForestClassifier(
            n_estimators=100,
            random_state=42
        )

    else:

        model = RandomForestRegressor(
            n_estimators=100,
            random_state=42
        )

    model.fit(X_train, y_train)

    return model