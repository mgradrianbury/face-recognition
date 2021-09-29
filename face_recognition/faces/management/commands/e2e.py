import numpy
from django.core.management import BaseCommand

from faces.models import FaceImageForTest, FaceLabel, FaceImage


class Command(BaseCommand):
    help = 'Check how good is validator'

    def add_arguments(self, parser):
        pass

    def handle(self, *args, **options):
        _DETECTION_THRESHOLD = 10.2

        test_image_with_parent = FaceImageForTest.objects.filter(label_id__parent__isnull=False)
        test_image_without_parent = FaceImageForTest.objects.filter(label_id__parent__isnull=True)

        embedding_array = numpy.array([f.embedding_array for f in FaceImage.objects.all()])

        result = []
        with_good_result = []
        for image in test_image_with_parent:
            embedding = image.embedding_array
            for_norm = embedding - embedding_array
            norm = numpy.linalg.norm(for_norm, axis=1)

            label_exists = True if numpy.min(norm) < _DETECTION_THRESHOLD else False
            result.append(label_exists)
            if label_exists is True:
                with_good_result.append(image.id)

        all_hits = len(test_image_with_parent)
        good_hits = numpy.sum(result)
        bad_hits = all_hits - good_hits

        self.stdout.write("Validator", style_func=self.style.SUCCESS)
        self.stdout.write("All images   : {}".format(all_hits), style_func=self.style.SUCCESS)
        self.stdout.write("Good predict : {}".format(good_hits), style_func=self.style.SUCCESS)
        self.stdout.write("Bad predict  : {}".format(bad_hits), style_func=self.style.SUCCESS)
        self.stdout.write("Mean         : {:0.1f}%".format(numpy.mean(result) * 100), style_func=self.style.SUCCESS)

        only_good_records = FaceImageForTest.objects.filter(id__in=with_good_result)

        idx = [image.label.label for image in only_good_records]
        predicted_labels = FaceLabel.predict_labels_for_embeddings(
            [test_image.embedding_array for test_image in only_good_records]
        )
        result = [idx[i] == predicted_labels[i] for i in range(len(idx))]

        all_hits = len(only_good_records)
        good_hits = numpy.sum(result)
        bad_hits = all_hits - good_hits

        self.stdout.write("Classification", style_func=self.style.SUCCESS)
        self.stdout.write("All images   : {}".format(all_hits), style_func=self.style.SUCCESS)
        self.stdout.write("Good predict : {}".format(good_hits), style_func=self.style.SUCCESS)
        self.stdout.write("Bad predict  : {}".format(bad_hits), style_func=self.style.SUCCESS)
        self.stdout.write("Mean         : {:0.1f}%".format((good_hits / all_hits) * 100), style_func=self.style.SUCCESS)
