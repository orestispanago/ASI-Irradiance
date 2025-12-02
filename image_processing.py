import numpy as np
import cv2
import logging


logger = logging.getLogger(__name__)


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
    logger.debug(f"Processed image: {fname}.")
