from typing import Callable

import cv2
import numpy as np

from filters import apply_filter, filters
from stickers import apply_sticker, stickers
from dynamic_stickers import apply_dynamic_sticker, dynamic_stickers


mouse_point = (None, None)

def get_mouse_point(event, x, y, flags, param):
    global mouse_point
    if event == cv2.EVENT_LBUTTONDBLCLK:
        mouse_point = (x, y)


def none(_frame, _=None):
    return _frame


# Webcam Capture
cap = cv2.VideoCapture(0)

# Define initial effects as none
_filter: Callable = none
_sticker: Callable = none
_dynamic_sticker: Callable = none


cv2.namedWindow("Trabalho GB - Luiz e Guilherme")
cv2.setMouseCallback('Trabalho GB - Luiz e Guilherme', get_mouse_point)

while True:
    try:
        res, frame = cap.read()

        # Apply current effects to frame
        frame = apply_filter(frame, _filter)
        frame = apply_sticker(frame, _sticker, mouse_point)
        frame = apply_dynamic_sticker(frame, _dynamic_sticker)

        # Read input
        key_ascii = cv2.waitKey(1)

        # default case
        if key_ascii == -1:
            pass

        else:
            key = chr(key_ascii)
            print(key)

            # quit
            if key == 'q':
                break

            # reset
            elif key == '-':
                _filter = none
                _sticker = none
                _dynamic_sticker = none
                mouse_point = (None, None)
                continue

            # filter
            else:
                if key in filters:
                    _filter = filters[key]

                if key in stickers and mouse_point[0]:
                    _sticker = stickers[key]

                if key in dynamic_stickers:
                    _dynamic_sticker = dynamic_stickers[key]

        cv2.imshow('Trabalho GB - Luiz e Guilherme', frame)
    except Exception as e:
        print(f'Error: {e}')
        break


cv2.release()
cv2.destroyAllWindows()
