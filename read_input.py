import pandas as pd


# taken from https://github.com/yypai/ITX-pandas/blob/master/itx_to_pandas_df.py
def itx_to_pandas(path: str) -> pd.DataFrame:
    file = open(path, 'r')
    a = file.read().splitlines()
    file.close()

    key = []
    values = []
    cur = 0
    while cur < len(a) - 1:

        if 'WAVES/D/N=' in a[cur]:
            key.append(a[cur].split('\'')[1].split('.')[0])
            loc_len = int(a[cur].split(')')[0].split('(')[1])
            value = a[cur + 2: cur + 2 + loc_len]
            value = [float(line) for line in value]
            values.append(value)
            cur += loc_len
            cur += 2

        cur += 1

    dfn = pd.DataFrame(values).T
    dfn.columns = key
    return dfn


def read_counts_file(path: str, max_row=2048) -> pd.Series:
    with open(path, "r") as f:
        raw_data = f.read().split("\n")
    numeric_data = [float(data) for data in raw_data if data.isnumeric()]

    data_df = pd.Series(numeric_data)

    return data_df.iloc[:max_row]


if __name__ == '__main__':
    df = read_counts_file("histogram.itx")
