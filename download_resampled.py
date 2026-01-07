import glob
import pandas as pd


resampled = glob.glob("remote-paths/resampled1min/*.csv")
for fname in resampled:
    df = pd.read_csv(fname, index_col="timestamp")
    remote_paths = df["remote_paths"].values.tolist()
