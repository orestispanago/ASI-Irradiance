import logging
import logging.config
import os
import traceback
import pandas as pd
import ftp_utils
from solar_calculations import calc_SZA
from file_utils import list_to_csv

os.chdir(os.path.dirname(os.path.abspath(__file__)))
logging.config.fileConfig("logging.conf", disable_existing_loggers=False)
logging.getLogger("PIL").setLevel(logging.INFO)
logging.getLogger("h5py").setLevel(logging.INFO)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
logger = logging.getLogger(__name__)


def select_paths_sza_less_than_80(all_paths):
    logger.debug(f"Selecting SZA < 80 from {len(all_paths)} files...")
    selected_paths = []
    for path in all_paths:
        base_name = os.path.basename(path)
        date_time = pd.to_datetime(
            base_name[:15], format="%Y%m%d_%H%M%S"
        ).tz_localize("UTC")
        sza = calc_SZA(date_time)
        if sza < 80:
            selected_paths.append(path)
    logger.debug(f"Selecting {len(selected_paths)} files where SZA < 80.")
    return selected_paths


def run_multiple_files():
    logger.info(f"{'-' * 15} START BULK RUN{'-' * 15}")
    remote_paths = ftp_utils.get_remote_paths()
    list_to_csv(remote_paths, "all_paths.csv")
    selected_paths = select_paths_sza_less_than_80(remote_paths)
    list_to_csv(selected_paths, "selected_paths.csv")
    ftp_utils.download_multiple(selected_paths)


if __name__ == "__main__":
    try:
        run_multiple_files()
    except:
        logger.error("uncaught exception: %s", traceback.format_exc())
