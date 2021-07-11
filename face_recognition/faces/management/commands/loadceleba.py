from io import BytesIO
from os import path
import cv2
import pandas as pd
from PIL import Image
from django.core.files import File
from django.core.management import BaseCommand
from faces.models import FaceImageForTest, FaceLabel, FaceLabelForTest, FaceImage


def _get_file(full_path: str, file_name: str):
    array = cv2.cvtColor(cv2.imread(full_path), cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(array)
    thumb_io = BytesIO()
    pil_image.save(thumb_io, 'JPEG')
    return File(thumb_io, name=file_name)


class Command(BaseCommand):
    help = 'Load faces from CelebA'

    def add_arguments(self, parser):
        parser.add_argument("--images",
                            help='Path to images',
                            type=str,
                            default="/home/adrian/MyCurse/master_thesis/faces/CelebA/Img/img_align_celeba")
        parser.add_argument("--identity",
                            help='Path to csv file',
                            type=str,
                            default="/home/adrian/MyCurse/master_thesis/faces/CelebA/Anno/identity_CelebA.txt")
        parser.add_argument("--faces", help='How many faces do you want to load', type=int, default=500)
        parser.add_argument("--extra", help='How many faces do you want to load without parent', type=int, default=100)

    def handle(self, *args, **options):
        identity_path = options['identity']
        images_path = options['images']
        faces_count = options['faces']
        faces_extra = options['extra']

        data = pd.read_csv(identity_path, header=None, sep=" ")
        ids = data[1].unique()

        first_n_faces = ids[:faces_count]
        extra_n_faces = ids[:faces_count + faces_extra]

        for person_id in first_n_faces:
            images_for_peron = data.loc[data[1] == person_id][0].array
            images_for_training = images_for_peron[0:5]

            label = FaceLabel(label=person_id)
            label.save()

            for file_name in images_for_training:
                full_path = path.join(images_path, file_name)

                try:
                    FaceImage(
                        original_image=_get_file(full_path=full_path, file_name=file_name),
                        label=label
                    ).save()
                except:
                    self.stdout.write(
                        msg="File {} can not be parsed".format(file_name),
                        style_func=self.style.WARNING
                    )

        for person_id in extra_n_faces:
            images_for_peron = data.loc[data[1] == person_id][0].array
            images_for_tests = images_for_peron[5:8]

            label_for_test = FaceLabelForTest(label=person_id, parent=FaceLabel.objects.filter(label=person_id).first())
            label_for_test.save()

            for file_name in images_for_tests:
                full_path = path.join(images_path, file_name)

                try:
                    FaceImageForTest(
                        original_image=_get_file(full_path=full_path, file_name=file_name),
                        label=label_for_test
                    ).save()
                except:
                    self.stdout.write(
                        msg="File {} can not be parsed".format(file_name),
                        style_func=self.style.WARNING
                    )
