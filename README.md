# ASI-Irradiance

* Downloads last image from FTP.

* Crops image and applies black mask to unused area.

* Calculates solar irradiance (DHI and GHI) from All Sky Image using pre-trained CNN model (saved in ```.h5``` files in ```models``` folder).

* Uploads predictions ```.csv``` to FTP. File contains a single wor with datetime, GHI and DHI predictions and is overwritten at each run.