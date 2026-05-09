from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, r2_score

def evaluate_model(model, X_test, y_test, task_type):

    predictions = model.predict(X_test)

    if task_type == "classification":

        return {
            "accuracy": accuracy_score(
                y_test,
                predictions
            ),
            "f1_score": f1_score(
                y_test,
                predictions,
                average="weighted"
            )
        }

    else:

        return {
            "mae": mean_absolute_error(
                y_test,
                predictions
            ),
            "r2_score": r2_score(
                y_test,
                predictions
            )
        }
    
def get_feature_importance( model, feature_names):

    if not hasattr(model, "feature_importances_"):
        return None

    importance = dict(
        zip(
            feature_names,
            model.feature_importances_
        )
    )

    importance = dict(
        sorted(
            importance.items(),
            key=lambda x: x[1],
            reverse=True
        )
    )

    return importance