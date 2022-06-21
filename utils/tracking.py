import cv2
import numpy as np


face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_eye.xml")


def detect_faces(frame):
    """
    Utiliza o modelo pré-treinado haarcascade para fazer a detecção das faces no frame
    Returns: Lista de faces detectadas
    """

    # Make image grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Initialize output vector
    all_faces = []

    # First, detect faces
    faces = face_cascade.detectMultiScale(gray, 1.2, 5)

    for (x, y, w, h) in faces:
        all_faces += [[x, y, w, h]]

    return all_faces


def detect_eyes(frame):
    """
    Utiliza o modelo pré-treinado haarcascade para fazer a detecção dos olhos no frame
    Returns: Lista dos olhos detectados
    """

    # Make image grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Initialize output vector
    all_eyes = []

    # First, detect faces
    faces = face_cascade.detectMultiScale(gray, 1.5, 5)

    for (x, y, w, h) in faces:
        face_region = gray[y:y + h, x:x + w]
        eyes = eye_cascade.detectMultiScale(face_region, 1.2, 5)
        for (ex, ey, ew, eh) in eyes:
            all_eyes += [[ex + x, ey + y, ew, eh]]

    return all_eyes
