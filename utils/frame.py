import numpy as np
from scipy.interpolate import UnivariateSpline


def lookup_table(x, y):
    spline = UnivariateSpline(x, y)
    return spline(range(256))


def alpha_compose(frame, sticker, x, y):
    """
    Implementação do algoritmo: Alpha compositing
    """
    frame_width = frame.shape[1]
    frame_height = frame.shape[0]

    if x >= frame_width or y >= frame_height:
        return frame

    h, w = sticker.shape[0], sticker.shape[1]

    if x + w > frame_width:
        w = frame_width - x
        sticker = sticker[:, :w]

    if y + h > frame_height:
        h = frame_height - y
        sticker = sticker[:h]

    if sticker.shape[2] < 4:
        sticker = np.concatenate(
            [
                sticker,
                np.ones((sticker.shape[0], sticker.shape[1], 1), dtype=sticker.dtype) * 255
            ],
            axis=2,
        )

    sticker_image = sticker[..., :3]
    mask = sticker[..., 3:] / 255.0
    frame[y:y+h, x:x+w] = (1.0 - mask) * frame[y:y+h, x:x+w] + mask * sticker_image
    return frame
