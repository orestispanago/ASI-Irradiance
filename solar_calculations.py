import pvlib
import pandas as pd

CAM_ALT = 24.78
CAM_LAT = 38.291381749413844
CAM_LON = 21.78860648659206


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
