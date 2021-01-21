from datetime import datetime

from matplotlib import pyplot
from numpy import load, expand_dims, ceil
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Normalizer
from sklearn.svm import SVC

if __name__ == '__main__':
    data = load('/home/adrian/MyCurse/master_thesis/embeddings.npz')
    trainX, trainy, testX, testy = data['arr_0'], data['arr_1'], data['arr_2'], data['arr_3']

    print('Dataset: train=%d, test=%d' % (trainX.shape[0], testX.shape[0]))

    in_encoder = Normalizer(norm='l2')
    trainX = in_encoder.transform(trainX)
    testX = in_encoder.transform(testX)

    data = load('/home/adrian/MyCurse/master_thesis/dataset.npz')
    testX_faces = data['arr_2']

    i = 1
    for face in testX_faces:
        pyplot.subplot(ceil(len(testX_faces) / 5), 5, i)
        pyplot.axis('off')
        pyplot.imshow(face)
        i += 1

    print("All faces ({})".format(len(testX_faces)))
    pyplot.show()

    # OK
    out_encoder = LabelEncoder()
    out_encoder.fit(trainy)
    trainy = out_encoder.transform(trainy)
    testy = out_encoder.transform(testy)

    model = SVC(kernel='linear', probability=True)
    model.fit(trainX, trainy)

    # yhat_train = model.predict(trainX)
    # yhat_test = model.predict(testX)
    #
    # score_train = accuracy_score(trainy, yhat_train)
    # score_test = accuracy_score(testy, yhat_test)
    #
    # print('Accuracy: train=%.3f, test=%.3f' % (score_train*100, score_test*100))

    selection = 33  # choice([i for i in range(testX.shape[0])])
    random_face_pixels = testX_faces[selection]
    random_face_emb = testX[selection]
    random_face_class = testy[selection]
    random_face_name = out_encoder.inverse_transform([random_face_class])

    start = datetime.now()
    samples = expand_dims(random_face_emb, axis=0)
    yhat_class = model.predict(samples)
    yhat_prob = model.predict_proba(samples)

    class_index = yhat_class[0]
    class_probability = yhat_prob[0, class_index] * 100
    predict_names = out_encoder.inverse_transform(yhat_class)
    end = datetime.now()

    print('Predicted: %s (%.3f)' % (predict_names[0], class_probability))
    print('Expected: %s' % random_face_name[0])
    print('Has been done in {}s'.format(end - start))

    pyplot.imshow(random_face_pixels)
    title = '%s (%.3f)' % (predict_names[0], class_probability)
    pyplot.title(title)
    pyplot.show()
