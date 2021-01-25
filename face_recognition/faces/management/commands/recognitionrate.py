import os

import numpy
from django.core.management import BaseCommand
from faces.models import FaceLabel


class Command(BaseCommand):
    help = 'Check recognition rate'

    def add_arguments(self, parser):
        parser.add_argument("--repeat", help='How many time do you want to repeat experiment', type=int, default=1)

    def handle(self, *args, **options):
        repeat = options['repeat']
        test_dir_path = '/home/adrian/MyCurse/master_thesis/faces/CelebA/Img/temp/test'

        labels = os.listdir(test_dir_path)
        labels_path = [os.path.join(test_dir_path, label) for label in labels]
        images_path = [os.path.join(base, os.listdir(base)[0]) for base in labels_path]

        results = [self._run_experiment(images_path, labels) for _ in range(repeat)]

        self.stdout.write("Mean : {: 6.2f}".format(numpy.mean(results)), style_func=self.style.SUCCESS)
        self.stdout.write("Std  : {: 6.2f}".format(numpy.std(results)), style_func=self.style.SUCCESS)

    def _run_experiment(self, paths, labels) -> float:
        predicted_labels = FaceLabel.predict_labels(paths)
        recognition_rate = 0
        for predicted_label, index in zip(predicted_labels, range(len(predicted_labels))):
            if labels[index] == predicted_label:
                recognition_rate = recognition_rate + 1

        calc = recognition_rate / len(labels)
        self.stdout.write("Experiment end with {: 6.2f} rate".format(calc))
        return calc
