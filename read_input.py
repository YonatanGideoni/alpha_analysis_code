import pandas as pd


def read_counts_file(path: str, max_row=2048) -> pd.Series:
    with open(path, "r") as f:
        raw_data = f.read().split("\n")
    numeric_data = [float(data) for data in raw_data if data.isnumeric()]

    data_df = pd.Series(numeric_data)

    return data_df.iloc[:max_row]


if __name__ == '__main__':
    df = read_counts_file("histogram.itx")
