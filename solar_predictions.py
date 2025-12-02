import logging
import pandas as pd
import keras
from solar_calculations import calc_ghi_clear

logger = logging.getLogger(__name__)


def predict_ghi_dhi(date_time, img_folder="img", dest_csv="ghi_dhi_result.csv"):
    test_generator = keras.src.legacy.preprocessing.image.ImageDataGenerator(
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
    kd_predictions = model_kd.predict(test_DHI_image, verbose=0)[0][0]
    kt_predictions = model_kt.predict(test_DHI_image, verbose=0)[0][0]
    ghi_clear = calc_ghi_clear(date_time)
    ghi_pred = kt_predictions * ghi_clear
    dhi_pred = kd_predictions * ghi_pred
    logger.debug(f"Predicted GHI: {ghi_pred}, DHI: {dhi_pred}.")
    result = {
        "Datetime_UTC": date_time,
        "GHI": ghi_pred,
        "DHI": dhi_pred,
    }
    pd.DataFrame([result]).to_csv(dest_csv, index=False)
    logger.debug(f"Saved predictions to: {dest_csv}.")
