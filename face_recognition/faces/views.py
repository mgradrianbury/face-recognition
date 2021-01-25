import os
from django.conf import settings
from django.core.files.base import ContentFile
from django.core.files.storage import default_storage
from django.http import HttpResponse, HttpRequest
from django.shortcuts import render
from django.views import View
from faces.models import FaceLabel


class FaceRecognition(View):
    def get(self, request: HttpRequest):
        return render(request, 'index.html')

    def post(self, request: HttpRequest):
        image = request.FILES.get("face")
        image_path = os.path.join(settings.MEDIA_ROOT, 'face_recognition', str(image))
        default_storage.save(image_path, ContentFile(image.read()))

        label_exists = FaceLabel.predict_if_label_exist(image_path)

        if label_exists is not True:
            return HttpResponse("Sorry. I do not know you.")

        predicted_label = FaceLabel.predict_label(image_path)
        default_storage.delete(image_path)

        return HttpResponse("I think you are {}.".format(predicted_label))
