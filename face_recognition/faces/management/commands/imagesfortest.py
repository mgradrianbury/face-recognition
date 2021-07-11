import os
import shutil
from io import BytesIO
from os import path
import cv2
import numpy
import pandas as pd
from PIL import Image
from django.core.files import File
from django.core.management import BaseCommand
from faces.models import FaceImageForTest, FaceLabel, FaceLabelForTest, FaceImage


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
        parser.add_argument("--copy_to",
                            help='Where put copied images',
                            type=str,
                            default="/home/adrian/MyCurse/master_thesis/faces/Test")

    def handle(self, *args, **options):
        identity_path = options['identity']
        images_path = options['images']
        copy_to = options['copy_to']

        data = pd.read_csv(identity_path, header=None, sep=" ")
        ids = data[1].unique()

        random_n_faces = numpy.random.choice(ids, 7)

        for person_id in random_n_faces:
            images_for_peron = data.loc[data[1] == person_id][0].array
            images_for_training = images_for_peron[0:5]

            for file_name in images_for_training:
                full_path = path.join(images_path, file_name)
                new_path = path.join(copy_to, str(person_id))
                os.makedirs(new_path, exist_ok=True)
                shutil.copyfile(full_path, path.join(new_path, file_name))
