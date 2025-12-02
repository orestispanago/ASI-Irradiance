import logging
import logging.config
import os
import traceback

import pandas as pd

import ftp_utils
from image_processing import crop_image_circle
from solar_calculations import calc_SZA, calc_ghi_clear
from solar_predictions import predict_ghi_dhi

os.chdir(os.path.dirname(os.path.abspath(__file__)))
logging.config.fileConfig("logging.conf", disable_existing_loggers=False)
logging.getLogger("PIL").setLevel(logging.INFO)
logging.getLogger("h5py").setLevel(logging.INFO)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
logger = logging.getLogger(__name__)


LOCAL_CSV = "ghi_dhi_result.csv"
REMOTE_CSV = f"{ftp_utils.FTP_DIR}/{LOCAL_CSV}"


def main():
    logger.info(f"{'-' * 15} START {'-' * 15}")
    last_remote_file = ftp_utils.get_last_file_path()
    base_name = os.path.basename(last_remote_file)
    date_time = pd.to_datetime(
        base_name[:15], format="%Y%m%d_%H%M%S"
    ).tz_localize("UTC")
    sza = calc_SZA(date_time)
    if sza < 80:
        local_path = f"img/{base_name}"
        ftp_utils.download(last_remote_file, local_path)
        crop_image_circle(local_path)
        predict_ghi_dhi(date_time)
        os.remove(local_path)
        ftp_utils.upload(LOCAL_CSV, REMOTE_CSV)
    else:
        logger.warning("sza < 80. Skipping...")
    logger.info(f"{'-' * 15} SUCCESS {'-' * 15}")


if __name__ == "__main__":
    try:
        main()
    except:
        logger.error("uncaught exception: %s", traceback.format_exc())
