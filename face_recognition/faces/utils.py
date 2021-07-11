import cv2
from PIL import Image
from numpy import asarray, ndarray
from numpy import expand_dims
from mtcnn import MTCNN
from tensorflow.python.keras.models import load_model

_KERAS_MODEL = None
_MTCNN_DETECTOR = None


class FaceNotFoundError(Exception):
    pass


def _get_keras_model():
    global _KERAS_MODEL
    if _KERAS_MODEL is None:
        _KERAS_MODEL = load_model('/home/adrian/MyCurse/master_thesis/facenet_keras.h5', compile=False)
    return _KERAS_MODEL


def _get_mtcnn_detector() -> MTCNN:
    global _MTCNN_DETECTOR
    if _MTCNN_DETECTOR is None:
        _MTCNN_DETECTOR = MTCNN()
    return _MTCNN_DETECTOR


def get_embedding(face_pixels: ndarray):
    face_pixels = face_pixels.astype('float32')
    mean, std = face_pixels.mean(), face_pixels.std()
    face_pixels = (face_pixels - mean) / std
    samples = expand_dims(face_pixels, axis=0)
    yhat = _get_keras_model().predict(samples)
    return yhat[0]


def extract_face(file_path, required_size=(160, 160)):
    pixels = _to_pixels(file_path)

    results = _get_mtcnn_detector().detect_faces(pixels)

    if len(results) == 0:
        raise FaceNotFoundError

    x1, y1, width, height = results[0]['box']
    x1, y1 = abs(x1), abs(y1)
    x2, y2 = x1 + width, y1 + height
    face = pixels[y1:y2, x1:x2]

    image = Image.fromarray(face)
    image = image.resize(required_size)
    return asarray(image)


def _to_pixels(path_to_file, resize_to=(350, 350)):
    cv2_image = cv2.cvtColor(cv2.imread(path_to_file), cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(cv2_image)
    pil_image.thumbnail(resize_to, Image.ANTIALIAS)
    return asarray(pil_image)
