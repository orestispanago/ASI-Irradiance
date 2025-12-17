import os
import logging
import pandas as pd

from solar_calculations import calc_SZA

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


def list_to_csv(the_list, fname):
    with open(fname, "w") as file:
        file.write("\n".join(the_list))
