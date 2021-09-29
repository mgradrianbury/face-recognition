from multiprocessing import Pool, cpu_count

import numpy
import pandas as pd
from django.core.management import BaseCommand
from sklearn.metrics.pairwise import euclidean_distances
from tqdm import tqdm

from faces.models import FaceImage, FaceLabelForTest, FaceImageForTest


def _run_experiment(all_images, test_image, threshold, good_answer):
    distances = [
        euclidean_distances([test_image.embedding_array], [face_image.embedding_array])
        for face_image in all_images
    ]
    return threshold, (True if numpy.min(distances) < threshold else False) == good_answer


class Command(BaseCommand):
    help = 'Check how good is validator on different threshold'

    def add_arguments(self, parser):
        pass

    def handle(self, *args, **options):
        test_labels = FaceLabelForTest.objects.all()
        all_images = FaceImage.objects.all()

        args = []

        for test_label in test_labels:
            for image in FaceImageForTest.objects.filter(label=test_label):
                for index in range(80, 120, 1):
                    threshold = index / 10
                    args.append(
                        (
                            all_images,
                            image,
                            threshold,
                            test_label.parent is not None
                        )
                    )

        with Pool(cpu_count()) as process:
            r = process.starmap(
                _run_experiment,
                tqdm(args)
            )

        df = pd.DataFrame(data=r, columns=("Threshold", "Answer"))
        df.to_csv('./validator_accuracy.csv')
