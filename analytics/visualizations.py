import matplotlib.pyplot as plt
import pandas as pd


def plot_missing_values(df):

    missing = df.isnull().sum()

    missing = missing[missing > 0]

    if missing.empty:

        fig, ax = plt.subplots(figsize=(6, 2))

        ax.text(
            0.5,
            0.5,
            "No Missing Values Detected",
            ha="center",
            va="center",
            fontsize=14
        )

        ax.axis("off")

        return fig

    fig, ax = plt.subplots(figsize=(8, 4))

    missing.sort_values(ascending=False).plot(
        kind="bar",
        ax=ax
    )

    ax.set_title("Missing Values Per Column")
    ax.set_ylabel("Count")

    plt.tight_layout()

    return fig


def plot_numeric_distributions(df):

    numeric_cols = df.select_dtypes(include="number").columns

    if len(numeric_cols) == 0:
        return None

    fig, ax = plt.subplots(figsize=(8, 4))

    df[numeric_cols].hist(
        ax=ax
    )

    plt.tight_layout()

    return fig

def plot_correlation_heatmap(corr_matrix):

    if corr_matrix is None:
        return None

    fig, ax = plt.subplots(figsize=(6, 5))

    im = ax.imshow(corr_matrix)

    ax.set_xticks(range(len(corr_matrix.columns)))
    ax.set_yticks(range(len(corr_matrix.columns)))

    ax.set_xticklabels(
        corr_matrix.columns,
        rotation=45,
        ha="right"
    )

    ax.set_yticklabels(corr_matrix.columns)

    plt.colorbar(im)

    plt.tight_layout()

    return fig

def plot_feature_importance(feature_importance):

    if not feature_importance:
        return None

    features = list(feature_importance.keys())[:10]
    values = list(feature_importance.values())[:10]

    fig, ax = plt.subplots(figsize=(8, 4))

    ax.bar(features, values)

    ax.set_title("Top Feature Importances")

    plt.xticks(rotation=45)

    plt.tight_layout()

    return fig