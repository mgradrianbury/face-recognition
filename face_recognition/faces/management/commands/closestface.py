import os

import numpy
from django.core.management import BaseCommand
from sklearn.metrics.pairwise import euclidean_distances

from faces.models import FaceImage, FaceLabel
from faces.utils import get_embedding, extract_face


class Command(BaseCommand):
    help = 'Check recognition rate'

    def add_arguments(self, parser):
        pass

    def handle(self, *args, **options):
        test_dir_path = '/home/adrian/MyCurse/master_thesis/faces/CelebA/Img/temp/test'

        labels = []
        images_path = []
        for label in os.listdir(test_dir_path):
            labels_path = os.path.join(test_dir_path, label)
            for image in os.listdir(labels_path):
                image_path = os.path.join(labels_path, image)
                images_path.append(image_path)
                labels.append(label)

        face_images = FaceImage.objects.all()

        predicted_labels = FaceLabel.predict_labels(images_path)

        min_distances = []
        hits = []
        for path, index in zip(images_path, range(len(images_path))):
            face = extract_face(path)
            embedding = get_embedding(face)

            distances = [
                euclidean_distances([embedding], [face_image.embedding_array])
                for face_image in face_images
            ]
            hits.append(predicted_labels[index] == labels[index])
            min_distances.append(numpy.min(distances))
            self.stdout.write("Min: {: 6.2f} Hit: {} ({} vs {})".format(
                    min_distances[-1], hits[-1], predicted_labels[index], labels[index]
                ))

        bad_hits = numpy.where(numpy.array(hits) == False)
        good_hits = numpy.where(numpy.array(hits) == True)

        bad_min = numpy.min(numpy.array(min_distances)[bad_hits])
        good_max = numpy.max(numpy.array(min_distances)[good_hits])

        self.stdout.write("Min value for BAD  hits is {: 6.2f}".format(bad_min), style_func=self.style.SUCCESS)
        self.stdout.write("Max value for GOOD hits is {: 6.2f}".format(good_max), style_func=self.style.SUCCESS)
