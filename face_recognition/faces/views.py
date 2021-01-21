import os

from django.conf import settings
from django.core.files.base import ContentFile
from django.core.files.storage import default_storage
from django.http import HttpResponse, HttpRequest
from django.shortcuts import render
from django.views import View
from numpy import expand_dims
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from faces.models import FaceLabel
from faces.utils import extract_face, get_embedding


def index(request: HttpRequest):
    return render(request, 'index.html')


class UploadFacesView(View):

    def get(self, request: HttpRequest):
        return render(request, 'upload.html')

    def post(self, request: HttpRequest):
        faces = request.FILES
        print(faces)
        return 'OK'


def result(request: HttpRequest):
    image = request.FILES.get("face")
    image_path = os.path.join(settings.MEDIA_ROOT, 'face_recognition', str(image))
    default_storage.save(image_path, ContentFile(image.read()))
    label, prob = _get_label(image_path)

    return HttpResponse("I think you are {}. I am sure for {}%.".format(label, int(prob)))


def _get_label(path_to_file):
    face_pixels = extract_face(path_to_file)
    embedding = get_embedding(face_pixels)

    train_x, train_y = FaceLabel.embeddings()

    out_encoder = LabelEncoder()
    out_encoder.fit(train_y)
    train_y = out_encoder.transform(train_y)

    model = SVC(kernel='linear', probability=True)
    model.fit(train_x, train_y)

    samples = expand_dims(embedding, axis=0)
    yhat_class = model.predict(samples)
    yhat_prob = model.predict_proba(samples)

    predict_names = out_encoder.inverse_transform(yhat_class)

    class_index = yhat_class[0]
    class_probability = yhat_prob[0, class_index] * 100

    return predict_names[0], class_probability
