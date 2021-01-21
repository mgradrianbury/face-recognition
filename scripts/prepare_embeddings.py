import keras
import numpy
import tensorflow as tf
from keras.models import load_model
from numpy import asarray
from numpy import expand_dims
from numpy import load
from numpy import savez_compressed


def get_embedding(model, face_pixels):
    face_pixels = face_pixels.astype('float32')
    mean, std = face_pixels.mean(), face_pixels.std()
    face_pixels = (face_pixels - mean) / std
    samples = expand_dims(face_pixels, axis=0)
    yhat = model.predict(samples)
    return yhat[0]


if __name__ == '__main__':
    print("tensorflow version    {}".format(tf.__version__))
    print("keras version         {}".format(keras.__version__))
    print("numpy version         {}".format(numpy.__version__))

    assert tf.__version__ == "2.3.0"
    assert keras.__version__ == "2.4.3"
    assert numpy.__version__ == "1.18.5"

    data = load('/home/adrian/MyCurse/master_thesis/dataset.npz')
    trainX, trainy, testX, testy = data['arr_0'], data['arr_1'], data['arr_2'], data['arr_3']

    print('Loaded: ', trainX.shape, trainy.shape, testX.shape, testy.shape)

    model = load_model('/home/adrian/MyCurse/master_thesis/facenet_keras.h5', compile=False)
    print('Loaded Model')

    newTrainX = list()

    for face_pixels in trainX:
        embedding = get_embedding(model, face_pixels)
        newTrainX.append(embedding)

    newTrainX = asarray(newTrainX)
    print(newTrainX.shape)

    print(len(newTrainX))

    newTestX = list()

    for face_pixels in testX:
        embedding = get_embedding(model, face_pixels)
        newTestX.append(embedding)

    newTestX = asarray(newTestX)
    print(newTestX.shape)

    savez_compressed('/home/adrian/MyCurse/master_thesis/embeddings.npz', newTrainX, trainy, newTestX, testy)
