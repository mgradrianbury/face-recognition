import numpy
from django.core.management import BaseCommand
from faces.models import FaceImageForTest, FaceLabel


class Command(BaseCommand):
    help = 'Check how good is face net recognition system'

    def add_arguments(self, parser):
        pass

    def handle(self, *args, **options):
        test_images = FaceImageForTest.objects.filter(label_id__parent__isnull=False)

        idx = [image.label.label for image in test_images]
        predicted_label = FaceLabel.predict_labels_for_embeddings(
            [test_image.embedding_array for test_image in test_images]
        )

        result = [idx[i] == predicted_label[i] for i in range(len(idx))]

        good_hits = numpy.sum(result)
        all_hits = len(idx)
        bad_hits = all_hits - good_hits

        self.stdout.write("All images   : {}".format(all_hits), style_func=self.style.SUCCESS)
        self.stdout.write("Good predict : {}".format(good_hits), style_func=self.style.SUCCESS)
        self.stdout.write("Bad predict  : {}".format(bad_hits), style_func=self.style.SUCCESS)
        self.stdout.write("Mean         : {:0.1f}%".format(numpy.mean(result) * 100), style_func=self.style.SUCCESS)
