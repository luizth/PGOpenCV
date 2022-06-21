import cv2


def rescale(frame, scale=0.75):
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)
    dimensions = (width, height)

    return cv2.resize(frame, dimensions, interpolation=cv2.INTER_AREA)


def resize(frame, width=500, height=500):
    frame.set(3, width)
    frame.set(4, height)


def rebright(frame, brightness=100):
    frame.set(10, brightness)


def rotate(frame, angle=0, rot_point=None):
    (height, width) = frame.shape[:2]

    if not rot_point:
        rot_point = (width//2, height//2)  # center

    rot_mat = cv2.getRotationMatrix2D(rot_point, angle, 1.0)
    dimensions = (width, height)
    return cv2.warpAffine(frame, rot_mat, dimensions)
