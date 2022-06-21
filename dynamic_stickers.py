from typing import Callable, Dict

import cv2

from utils.frame import alpha_compose
from utils.tracking import detect_faces, detect_eyes

eye_img = cv2.imread("resources/eye.png", -1)
disguise_img = cv2.imread("resources/disguise.png", -1)


# Utils
def apply_dynamic_sticker(_frame, dynamic_sticker: Callable):
    return dynamic_sticker(_frame)


# Dynamic Stickers
def sleep(frame):
    # Get tracking of eyes
    eyes = detect_eyes(frame)

    for (ex, ey, ew, eh) in eyes:
        dynamic_sticker = cv2.resize(eye_img, (ew, eh))
        frame = alpha_compose(frame, dynamic_sticker, ex, ey)

    return frame


def disguise(frame):
    # Get tracking of faces
    faces = detect_faces(frame)

    for (x, y, w, h) in faces:
        dynamic_sticker = cv2.resize(disguise_img, (w, h))
        frame = alpha_compose(frame, dynamic_sticker, x, y)

    return frame


dynamic_stickers: Dict[str, Callable] = {
    'x': sleep,
    'y': disguise
}
