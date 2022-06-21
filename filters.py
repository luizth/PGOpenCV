from typing import Callable, Dict, List

import cv2
import numpy as np

from utils.frame import lookup_table
from utils.transform import rotate


# Utils
def apply_filter(_frame, filter: Callable):
    return filter(_frame)


# Filters
def greyscale(frame):
    frame_greyscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return frame_greyscale


def blur(frame, kernel=(15, 15)):
    frame_blur = cv2.GaussianBlur(frame, kernel, 0)
    return frame_blur


def bright(frame, beta=100):
    frame_bright = cv2.convertScaleAbs(frame, beta=beta)
    return frame_bright


def edge_detection(frame, x=100):
    frame_canny = cv2.Canny(frame, x, x)
    return frame_canny


def emboss_effect(frame):
    kernel = np.array([[0, -1, -1], [1, 0, -1], [1, 1, 0]])
    frame_emboss = cv2.filter2D(frame, -1, kernel)
    return frame_emboss


def sepia(frame):
    frame_sepia = np.array(frame, dtype=np.float64)
    frame_sepia = cv2.transform(frame_sepia, np.matrix([[0.272, 0.534, 0.131], [0.349, 0.686, 0.168], [0.393, 0.769, 0.189]]))
    frame_sepia[np.where(frame_sepia > 255)] = 255  # normalizing values greater than 255
    frame_sepia = np.array(frame_sepia, dtype=np.uint8)
    return frame_sepia


def pencil_sketch_grey(frame):
    sk_gray, sk_color = cv2.pencilSketch(frame, sigma_s=60, sigma_r=0.07, shade_factor=0.1)
    return sk_gray


def invert(frame):
    frame_inv = cv2.bitwise_not(frame)
    return frame_inv


def summer(frame):
    increase = lookup_table([0, 64, 128, 256], [0, 80, 160, 256])
    blue_channel, green_channel, red_channel = cv2.split(frame)

    red_channel = cv2.LUT(red_channel, increase).astype(np.uint8)

    frame_summer = cv2.merge((blue_channel, green_channel, red_channel))
    return frame_summer


def winter(frame):
    increase = lookup_table([0, 64, 128, 256], [0, 80, 160, 256])
    blue_channel, green_channel, red_channel = cv2.split(frame)

    blue_channel = cv2.LUT(blue_channel, increase).astype(np.uint8)

    frame_winter = cv2.merge((blue_channel, green_channel, red_channel))
    return frame_winter


def nature(frame):
    increase = lookup_table([0, 64, 128, 256], [0, 80, 160, 256])
    blue_channel, green_channel, red_channel = cv2.split(frame)

    green_channel = cv2.LUT(green_channel, increase).astype(np.uint8)

    frame_nature = cv2.merge((blue_channel, green_channel, red_channel))
    return frame_nature


def upside_down(frame):
    return rotate(frame, 180)


filters: Dict[str, Callable] = {
    '0': greyscale,
    '1': blur,
    '2': edge_detection,
    '3': emboss_effect,
    '4': sepia,
    '5': summer,
    '6': winter,
    '7': nature,
    '8': pencil_sketch_grey,
    '9': invert,
    'u': upside_down
}
