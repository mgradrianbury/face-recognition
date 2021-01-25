from django.urls import path
from . import views

urlpatterns = [
    path('', views.FaceRecognition.as_view(), name="face_recognition"),
]
