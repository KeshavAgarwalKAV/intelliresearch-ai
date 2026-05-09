import pandas as pd

def load_dataset(file):

    try:

        df = pd.read_csv(file)

        return df

    except Exception as e:

        print(f"Dataset loading error: {e}")

        return None