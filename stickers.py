from typing import Callable, Dict

import cv2
import numpy as np

from utils.frame import alpha_compose


heart_img = cv2.imread("resources/heart.png", -1)
deal_with_it_img = cv2.imread("resources/deal-with-it.png", -1)
cactus_img = cv2.imread("resources/cactus.png", -1)
ronaldinho_img = cv2.imread("resources/ronaldinho.png", -1)
curling_img = cv2.imread("resources/curling.png", -1)


# Utils
def apply_sticker(_frame, sticker: Callable, point=(0, 0)):
    return sticker(_frame, point)


# Stickers
def heart(frame, point=(0, 0)):
    x, y = point
    sticker = cv2.resize(heart_img, (280, 280))
    return alpha_compose(frame, sticker, x, y)


def deal_with_it(frame, point=(0, 0)):
    x, y = point
    sticker = cv2.resize(deal_with_it_img, (280, 180))
    return alpha_compose(frame, sticker, x, y)


def cactus(frame, point=(0, 0)):
    x, y = point
    sticker = cv2.resize(cactus_img, (280, 180))
    return alpha_compose(frame, sticker, x, y)


def ronaldinho(frame, point=(0, 0)):
    x, y = point
    sticker = cv2.resize(ronaldinho_img, (230, 370))
    return alpha_compose(frame, sticker, x, y)


def curling(frame, point=(0, 0)):
    x, y = point
    sticker = cv2.resize(curling_img, (280, 280))
    return alpha_compose(frame, sticker, x, y)


stickers: Dict[str, Callable] = {
    'a': heart,
    'b': deal_with_it,
    'c': cactus,
    'd': ronaldinho,
    'e': curling,
}
