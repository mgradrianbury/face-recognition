from io import BytesIO
from os import listdir, path

import cv2
from PIL import Image
from django.core.files import File
from django.core.management import BaseCommand

from faces.models import FaceLabel, FaceImage


class Command(BaseCommand):
    help = 'Load faces from directory'

    def add_arguments(self, parser):
        pass

    def handle(self, *args, **options):
        train_path = "/home/adrian/MyCurse/master_thesis/faces/CelebA/Img/temp/train"

        for directory in listdir(train_path):
            images_path = path.join(train_path, directory)
            face_label = FaceLabel(label=directory)
            face_label.save()
            for image in listdir(images_path):
                image_path = path.join(images_path, image)

                try:
                    array = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
                    pil_image = Image.fromarray(array)
                    thumb_io = BytesIO()
                    pil_image.save(thumb_io, 'JPEG')
                    face = File(thumb_io, name=image)

                    face_image = FaceImage(
                        original_image=face,
                        label=face_label
                    )
                    face_image.save()
                except IndexError:
                    self.stdout.write(
                        msg="File {} for label {} can not be parsed".format(image, directory),
                        style_func=self.style.WARNING
                    )

            self.stdout.write("Label {} loaded with {} images".format(face_label.label, len(listdir(images_path))))

        self.stdout.write(
            'Successfully loaded faces',
            style_func=self.style.SUCCESS
        )
