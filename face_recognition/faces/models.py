import os
import numpy
from typing import List
from io import BytesIO
from PIL import Image
from django.conf import settings
from django.core.files import File
from django.core.files.storage import default_storage
from django.db import models
from sklearn.metrics import euclidean_distances
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from faces.utils import extract_face, get_embedding


class _BaseImage(models.Model):
    original_image = models.ImageField(max_length=200)
    face_image = models.ImageField(max_length=250, editable=False)
    embedding = models.BinaryField(editable=False)

    _EMBEDDING_TYPE = numpy.float32

    @property
    def embedding_array(self) -> List:
        return numpy.frombuffer(self.embedding, dtype=self._EMBEDDING_TYPE)

    def save(self, force_insert=False, force_update=False, using=None, update_fields=None):
        original_image_path = self.get_system_original_image_path()
        extracted_face_array = extract_face(original_image_path)
        default_storage.delete(original_image_path)

        image = Image.fromarray(extracted_face_array)
        thumb_io = BytesIO()
        image.save(thumb_io, 'JPEG')
        self.face_image = File(thumb_io, name=self._get_face_image_name())
        self.embedding = get_embedding(extracted_face_array).tobytes()

        super().save(force_insert, force_update, using, update_fields)

    def get_system_original_image_path(self) -> str:
        storage_path = os.path.join(settings.MEDIA_ROOT, 'face_to_crop', str(self.original_image.name))
        relative_image_storage_path = default_storage.save(storage_path, self.original_image)
        return os.path.join(settings.MEDIA_ROOT, relative_image_storage_path)

    def _get_face_image_name(self) -> str:
        return "{0}_{2}{1}".format(*os.path.splitext(self.original_image.name) + ('face',))

    class Meta:
        abstract = True


class FaceLabel(models.Model):
    label = models.SlugField(max_length=200)

    _DETECTION_THRESHOLD = 10.2

    @staticmethod
    def get_embeddings():
        """
        Return x and y where x is embedding array, y is label
        """
        x, y = list(), list()
        face_labels = FaceLabel.objects.all()
        for face_label in face_labels:
            face_images = face_label.face_images.all()
            labels = [face_label.label for _ in range(len(face_images))]
            embedding = [f.embedding_array for f in face_images]

            x.extend(embedding)
            y.extend(labels)

        return numpy.asarray(x), numpy.asarray(y)

    @staticmethod
    def predict_if_label_exist(path_to_image) -> bool:
        face = extract_face(path_to_image)
        embedding = get_embedding(face)
        image_faces = FaceImage.objects.all()
        distances = [
            euclidean_distances([embedding], [f.embedding_array])
            for f in image_faces
        ]

        return True if numpy.min(distances) < FaceLabel._DETECTION_THRESHOLD else False

    @staticmethod
    def predict_labels(path_to_images: List[str]) -> numpy.array:
        embeddings = [get_embedding(extract_face(f)) for f in path_to_images]
        return FaceLabel.predict_labels_for_embeddings(embeddings)

    @staticmethod
    def predict_labels_for_embeddings(embeddings: List) -> numpy.array:
        train_x, train_y = FaceLabel.get_embeddings()

        out_encoder = LabelEncoder()
        out_encoder.fit(train_y)
        train_y = out_encoder.transform(train_y)

        model = SVC(kernel='linear')
        model.fit(train_x, train_y)

        predicted_class = model.predict(embeddings)

        return out_encoder.inverse_transform(predicted_class)

    @staticmethod
    def predict_label(path_to_images: str) -> numpy.array:
        return FaceLabel.predict_labels([path_to_images])[0]

    def __str__(self):
        return self.label


class FaceImage(_BaseImage):
    label = models.ForeignKey(FaceLabel, on_delete=models.CASCADE, related_name='face_images')


class FaceLabelForTest(models.Model):
    parent = models.ForeignKey(FaceLabel, null=True, blank=True, on_delete=models.SET_NULL)
    label = models.SlugField(max_length=200)

    def __str__(self):
        return self.label


class FaceImageForTest(_BaseImage):
    label = models.ForeignKey(FaceLabelForTest, on_delete=models.CASCADE, related_name='face_images')
