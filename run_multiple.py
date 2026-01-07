import logging
import logging.config
import os
import glob
import traceback
import pandas as pd
import ftp_utils
from image_processing import crop_image_circle
from solar_calculations import calc_SZA
from file_utils import list_to_csv, select_paths_sza_less_than_80
from solar_predictions import predict_ghi_dhi

os.chdir(os.path.dirname(os.path.abspath(__file__)))
logging.config.fileConfig("logging.conf", disable_existing_loggers=False)
logging.getLogger("PIL").setLevel(logging.INFO)
logging.getLogger("h5py").setLevel(logging.INFO)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
logger = logging.getLogger(__name__)


def select_remote_paths(year, month):
    remote_dir = f"/cams/All-Sky/lapup/{year}/{month}"
    remote_paths = ftp_utils.get_remote_paths(ftp_dir=remote_dir)
    list_to_csv(remote_paths, f"remote-paths/all/all_{year}_{month}.csv")
    selected_paths = select_paths_sza_less_than_80(remote_paths)
    list_to_csv(
        selected_paths, f"remote-paths/selected/selected_{year}_{month}.csv"
    )
    # ftp_utils.download_multiple(selected_paths)


def resample_1min(df):
    df["timestamp"] = pd.to_datetime(
        df["remote_paths"], format="%Y%m%d_%H%M%S", exact=False, utc=True
    )
    df.set_index("timestamp", inplace=True)
    df = df.resample("1min").first()
    df = df.dropna()
    return df


def download_resampled1min(fname):
    df = pd.read_csv(fname, index_col="timestamp")
    logger.debug(f"Downloading {len(df)} files for: {fname}")
    resampled_paths = df["remote_paths"].values.tolist()
    ftp_utils.download_multiple(resampled_paths)
    logger.debug(f"Downloaded {len(df)} files for: {fname}")


def run_multiple_files():
    logger.info(f"{'-' * 15} START BULK RUN{'-' * 15}")
    # year = 2023
    # for m in range(8, 13):
    #     month = f"{m:02}"
    #     select_remote_paths(year, month)
    resampled = glob.glob("remote-paths/resampled1min/*.csv")
    for fname in resampled:
        download_resampled1min(fname)
        year_month = os.path.basename(fname)[-11:-4]
        local_paths = glob.glob("cams/**/*.jpg", recursive=True)
        results = []
        for local_path in local_paths:
            base_name = os.path.basename(local_path)
            date_time = pd.to_datetime(
                base_name[:15], format="%Y%m%d_%H%M%S"
            ).tz_localize("UTC")
            crop_image_circle(local_path)
            result = predict_ghi_dhi(date_time)
            results.append(result)
            os.remove(local_path)
        df_all = pd.concat(results)
        predictions_csv = f"predictions/predictions_{year_month}.csv"
        df_all.to_csv(predictions_csv, index=False)
        logger.info(f'Saved {len(df_all)} rows to "{predictions_csv}"')
    logger.info(f"{'-' * 15} SUCCESS {'-' * 15}")


if __name__ == "__main__":
    try:
        run_multiple_files()
    except:
        logger.error("uncaught exception: %s", traceback.format_exc())
