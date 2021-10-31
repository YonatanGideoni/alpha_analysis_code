import pandas as pd
import os

# Import Module


# Folder Path


def read_counts_file(path: str, max_row=2048) -> pd.Series:
    with open(path, "r") as f:
        raw_data = f.read().split("\n")
    numeric_data = [float(data) for data in raw_data if data.isnumeric()]
    start_time = [float(data[13:24]) for data in raw_data if data.startswith('StartTime')][0]
    end_time = [float(data[12:23]) for data in raw_data if data.startswith('StopTime')][0]
    data_df = pd.Series(numeric_data)


    return data_df.iloc[:max_row]


def read_counts_file_time(path: str, max_row=2048) -> pd.Series:
    with open(path, "r") as f:
        raw_data = f.read().split("\n")
    start_time = [float(data[13:24]) for data in raw_data if data.startswith('StartTime')][0]
    end_time = [float(data[12:23]) for data in raw_data if data.startswith('StopTime')][0]
    delta_t = int(end_time - start_time)

    return delta_t


if __name__ == '__main__':
    print('h')
    #df = read_counts_file("thr45measurement1104.itx")
    #delta_t = read_counts_file_time("thr45measurement1104.itx")
    #print(delta_t)