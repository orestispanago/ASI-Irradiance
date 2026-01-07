import glob
import pandas as pd

csv_files = glob.glob("remote-paths/selected/*.csv")


def resample_1min(df):
    df["timestamp"] = pd.to_datetime(
        df["remote_paths"], format="%Y%m%d_%H%M%S", exact=False, utc=True
    )
    df.set_index("timestamp", inplace=True)
    df = df.resample("1min").first()
    df = df.dropna()
    return df


for fname in csv_files:
    print(fname)
    df = pd.read_csv(fname, names=["remote_paths"])
    dest_fname = fname.replace("selected", "resampled1min")
    resampled1min = resample_1min(df)
    resampled1min.to_csv(dest_fname)
    # print(f"Saved {len(resampled1min)} rows in {dest_fname}")
