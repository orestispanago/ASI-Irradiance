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


def select_and_download():
    remote_paths = ftp_utils.get_remote_paths()
    list_to_csv(remote_paths, "all_paths.csv")
    selected_paths = select_paths_sza_less_than_80(remote_paths)
    list_to_csv(selected_paths, "selected_paths.csv")
    # ftp_utils.download_multiple(selected_paths)


def run_multiple_files():
    logger.info(f"{'-' * 15} START BULK RUN{'-' * 15}")
    select_and_download()
    # local_paths = glob.glob("cams/**/*.jpg", recursive=True)
    # results = []
    # for local_path in local_paths:
    #     base_name = os.path.basename(local_path)
    #     date_time = pd.to_datetime(
    #         base_name[:15], format="%Y%m%d_%H%M%S"
    #     ).tz_localize("UTC")
    #     crop_image_circle(local_path)
    #     result = predict_ghi_dhi(date_time)
    #     results.append(result)
    #     os.remove(local_path)
    # df_all = pd.concat(results)
    # df_all.to_csv("predictions.csv", index=False)
    # logger.info(f'Saved {len(df_all)} rows to "predictions.csv"')
    logger.info(f"{'-' * 15} SUCCESS {'-' * 15}")


if __name__ == "__main__":
    try:
        run_multiple_files()
    except:
        logger.error("uncaught exception: %s", traceback.format_exc())
