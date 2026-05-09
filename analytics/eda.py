def generate_dataset_summary(df):

    summary = {}

    summary["rows"] = df.shape[0]
    summary["columns"] = df.shape[1]

    summary["column_names"] = list(df.columns)

    summary["missing_values"] = (df.isnull().sum().to_dict())

    summary["dtypes"] = (df.dtypes.astype(str).to_dict())

    summary["numeric_columns"] = list(df.select_dtypes(include="number").columns)

    summary["categorical_columns"] = [
        col for col in df.columns
        if (
            df[col].dtype == "object"
            or df[col].nunique() <= 10
        )
    ]

    summary["numeric_summary"] = (df.describe().round(2).to_dict())

    return summary

def compute_correlations(df):

    numeric_df = df.select_dtypes(include="number")

    if numeric_df.shape[1] < 2:
        return None

    corr_matrix = numeric_df.corr()

    return corr_matrix