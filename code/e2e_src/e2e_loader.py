import os
import pandas as pd

def clean_dataframe(df):
    df = df.dropna()
    df = df[df["text"].str.len() > 0]
    cols = df.columns.to_list()
    if "label" in cols:
        df = df[df["label"].str.len() > 0]
    return df


def get_e2e_df(dataset_path):
    DATA_DIR = "../../data/e2e/"
    df = pd.read_csv(os.path.join(DATA_DIR, dataset_path))
    # df = df.rename(columns={"mr":"text", "ref":"label"})
    df = df.rename(columns={"mr":"text", "ref":"label", "MR":"text"},errors='ignore')
    return df