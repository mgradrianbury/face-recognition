import pandas as pd
from django.core.management import BaseCommand
from numpy import concatenate
from sklearn.metrics import euclidean_distances
from tqdm import tqdm

from faces.models import FaceImage


class Command(BaseCommand):
    help = 'Save distances between the same and others labels'

    def add_arguments(self, parser):
        pass

    def handle(self, *args, **options):

        distances = []

        for image in tqdm(FaceImage.objects.all()):
            others = FaceImage.objects.exclude(label_id__in=[image.label_id]).order_by('?')[:3]
            the_same = FaceImage.objects.filter(label_id=image.label_id).exclude(id__in=[image.id])

            for another_image in concatenate((others, the_same), axis=None):
                distance = euclidean_distances([image.embedding_array], [another_image.embedding_array])
                distances.append([
                    image.label_id,
                    image.id,
                    another_image.label_id,
                    another_image.id,
                    distance[0][0]]
                )

        df = pd.DataFrame(distances, columns=['image_one', 'image_one_id', 'image_two', 'image_two_id', 'distance'])
        df.to_csv('./distances.csv')
