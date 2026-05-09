from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


def preprocess_dataset(df, target_column):

    df = df.copy()

    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Encode categorical columns
    for col in X.columns:

        if X[col].dtype == "object":

            le = LabelEncoder()

            X[col] = le.fit_transform(
                X[col].astype(str)
            )

    # Encode target if categorical
    if y.dtype == "object":

        le = LabelEncoder()

        y = le.fit_transform(
            y.astype(str)
        )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    return X_train, X_test, y_train, y_test