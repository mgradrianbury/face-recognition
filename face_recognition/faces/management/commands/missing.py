import numpy
from django.core.management import BaseCommand
from sklearn.metrics.pairwise import euclidean_distances
from faces.models import FaceImage, FaceLabelForTest, FaceImageForTest
import pandas as pd
from tqdm import tqdm


class Command(BaseCommand):
    help = 'Check distance for missing images'

    def add_arguments(self, parser):
        pass

    def handle(self, *args, **options):
        face_label_without_parent = FaceLabelForTest.objects.filter(parent__isnull=True)

        min_list = []
        count = len(face_label_without_parent)
        progress_bar = tqdm(total=count)

        for index, test_label in zip(range(count), face_label_without_parent):
            known_images = FaceImage.objects.all()
            for image in FaceImageForTest.objects.filter(label=test_label):
                distances = [
                    euclidean_distances([image.embedding_array], [face_image.embedding_array])
                    for face_image in known_images
                ]
                min_list.append(numpy.min(distances))
            progress_bar.update(1)

        progress_bar.close()

        df = pd.DataFrame(data={'distance': min_list})
        df.to_csv('./missing.csv')
