import os
import numpy
from io import BytesIO
from PIL import Image
from django.conf import settings
from django.core.files import File
from django.core.files.storage import default_storage
from django.db import models
from faces.utils import extract_face, get_embedding


class FaceLabel(models.Model):
    label = models.SlugField(max_length=200)

    @staticmethod
    def embeddings():
        x, y = list(), list()
        face_labels = FaceLabel.objects.all()
        for face_label in face_labels:
            face_images = face_label.face_images.all()
            labels = [face_label.label for _ in range(len(face_images))]
            embedding = [f.embedding_array for f in face_images]

            x.extend(embedding)
            y.extend(labels)

        return numpy.asarray(x), numpy.asarray(y)

    def __str__(self):
        return self.label


class FaceImage(models.Model):
    label = models.ForeignKey(FaceLabel, on_delete=models.CASCADE, related_name='face_images')
    original_image = models.ImageField(max_length=200)
    face_image = models.ImageField(max_length=250, editable=False)
    embedding = models.BinaryField(editable=False)

    _EMBEDDING_TYPE = numpy.float32

    @property
    def embedding_array(self):
        return numpy.frombuffer(self.embedding, dtype=self._EMBEDDING_TYPE)

    def save(self, force_insert=False, force_update=False, using=None, update_fields=None):
        original_image_path = self._get_original_image_path()
        extracted_face_array = extract_face(original_image_path)
        default_storage.delete(original_image_path)

        image = Image.fromarray(extracted_face_array)
        thumb_io = BytesIO()
        image.save(thumb_io, 'JPEG')
        self.face_image = File(thumb_io, name=self._get_face_image_name())
        self.embedding = get_embedding(extracted_face_array).tobytes()

        super().save(force_insert, force_update, using, update_fields)

    def _get_original_image_path(self) -> str:
        storage_path = os.path.join(settings.MEDIA_ROOT, 'face_to_crop', str(self.original_image.name))
        relative_image_storage_path = default_storage.save(storage_path, self.original_image)
        return os.path.join(settings.MEDIA_ROOT, relative_image_storage_path)

    def _get_face_image_name(self):
        return "{0}_{2}{1}".format(*os.path.splitext(self.original_image.name) + ('face',))
