import os

import cv2
import keras
import numpy as np
import pandas as pd
import pvlib

import ftp_utils

CAM_ALT = 24.78
CAM_LAT = 38.291381749413844
CAM_LON = 21.78860648659206
LOCAL_CSV = "ghi_dhi_predictions.csv"
REMOTE_CSV = f"{ftp_utils.FTP_DIR}/{LOCAL_CSV}"


def crop_image_circle(fname):
    img = cv2.imread(fname)

    height, width, cols = img.shape
    center_x_offset = 15
    center_y = int(height / 2)
    center_x = int(width / 2) + center_x_offset

    radius_offset = 10
    radius = center_y + radius_offset

    mask = np.zeros_like(img)
    mask = cv2.circle(mask, (center_x, center_y), radius, (255, 255, 255), -1)
    result = cv2.bitwise_and(img, mask)
    crop = result[0:height, center_x - radius : center_x - radius + 2 * radius]

    cv2.imwrite("img/cropped/preproc.png", crop)


def calc_SZA(date_time):
    solar_position = pvlib.solarposition.get_solarposition(
        date_time,
        latitude=CAM_LAT,
        longitude=CAM_LON,
        altitude=CAM_ALT,
    )
    return solar_position["apparent_zenith"].values[0]


def calc_ghi_clear(date_time):
    apparent_zenith = calc_SZA(date_time)
    airmass_rel = pvlib.atmosphere.get_relative_airmass(apparent_zenith)
    pressure = pvlib.atmosphere.alt2pres(CAM_ALT)
    airmass_abs = pvlib.atmosphere.get_absolute_airmass(airmass_rel, pressure)
    linke_turbidity = pvlib.clearsky.lookup_linke_turbidity(
        pd.DatetimeIndex([date_time]),
        latitude=CAM_LAT,
        longitude=CAM_LON,
    )
    dni_extra = pvlib.irradiance.get_extra_radiation(date_time)
    clear_sky_ineichen = pvlib.clearsky.ineichen(
        apparent_zenith,
        airmass_abs,
        linke_turbidity,
        CAM_ALT,
        dni_extra,
    )
    return clear_sky_ineichen["ghi"].values[0]


def predict_ghi_dhi(date_time, img_folder="img"):
    test_generator = keras.preprocessing.image.ImageDataGenerator(
        rescale=1.0 / 255
    )
    test_DHI_image = test_generator.flow_from_directory(
        img_folder,
        target_size=(128, 128),
        class_mode=None,
        color_mode="rgb",
        shuffle=False,
    )
    model_kd = keras.models.load_model("models/SCNN_Kd_model.h5")
    model_kt = keras.models.load_model("models/SCNN_Kt_model.h5")
    kd_predictions = model_kd.predict(test_DHI_image)[0][0]
    kt_predictions = model_kt.predict(test_DHI_image)[0][0]
    ghi_clear = calc_ghi_clear(date_time)
    ghi_pred = kt_predictions * ghi_clear
    dhi_pred = kd_predictions * ghi_pred
    result = {
        "Datetime_UTC": date_time,
        "GHI_pred": ghi_pred,
        "DHI_pred": dhi_pred,
    }
    pd.DataFrame([result]).to_csv(LOCAL_CSV, index=False)


def main():
    last_remote_file = ftp_utils.get_last_file_path()
    base_name = os.path.basename(last_remote_file)
    date_time = pd.to_datetime(
        base_name[:15], format="%Y%m%d_%H%M%S"
    ).tz_localize("UTC")
    sza = calc_SZA(date_time)
    if sza < 100:
        local_path = f"img/{base_name}"
        ftp_utils.download(last_remote_file, local_path)
        crop_image_circle(local_path)
        predict_ghi_dhi(date_time)
        print("predicted")
        os.remove(local_path)
        ftp_utils.upload(LOCAL_CSV, REMOTE_CSV)
    else:
        print("sza < 80. Not predicted.")


main()
