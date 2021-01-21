import time
from math import ceil
from os import listdir, path
import cv2
import mtcnn
from PIL import Image
from mtcnn.mtcnn import MTCNN
from numpy import asarray, savez_compressed

detector = MTCNN()


def prepare_image(path_to_image, resize_to=(350, 350)):
    cv2_image = cv2.cvtColor(cv2.imread(path_to_image), cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(cv2_image)
    pil_image.thumbnail(resize_to, Image.ANTIALIAS)
    return asarray(pil_image)


def extract_face(path_to_image, required_size=(160, 160)):
    file_name = path.basename(path_to_image)
    temp_path = '/home/adrian/MyCurse/master_thesis/faces/temp/{}'
    file_temp = temp_path.format(file_name)

    if path.isfile(file_temp.format(file_name)):
        image = Image.open(file_temp)
    else:
        pixels = prepare_image(path_to_image)
        results = detector.detect_faces(pixels)
        x1, y1, width, height = results[0]['box']
        x1, y1 = abs(x1), abs(y1)
        x2, y2 = x1 + width, y1 + height
        face = pixels[y1:y2, x1:x2]
        image = Image.fromarray(face)
        image = image.resize(required_size)

        image.save(file_temp.format(file_name))

    return asarray(image)


def load_faces(directory):
    faces = list()
    for filename in listdir(directory):
        src = path.join(directory, filename)

        try:
            face = extract_face(src)
        except Exception as e:
            print("Cannot extract face for {}".format(filename))
            continue

        faces.append(face)
    return faces


def load_dataset(directory):
    x, y = list(), list()
    for subdir in listdir(directory):
        src = path.join(directory, subdir)

        if not path.isdir(src):
            continue

        faces = load_faces(src)
        labels = [subdir for _ in range(len(faces))]

        x.extend(faces)
        y.extend(labels)
    return asarray(x), asarray(y)


if __name__ == '__main__':
    print("mtcnn version         {}".format(mtcnn.__version__))
    assert "0.1.0" == mtcnn.__version__

    t0 = time.time()
    trainX, trainy = load_dataset('/home/adrian/MyCurse/master_thesis/faces/train')
    testX, testy = load_dataset('/home/adrian/MyCurse/master_thesis/faces/val')

    savez_compressed('/home/adrian/MyCurse/master_thesis/dataset.npz', trainX, trainy, testX, testy)
    t1 = time.time()

    print('Time: {}'.format(ceil(t1 - t0)))
