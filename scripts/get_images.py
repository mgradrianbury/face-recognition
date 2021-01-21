import os
from shutil import copyfile
from os import path
import pandas as pd

data = pd.read_csv('/home/adrian/Downloads/zdjęcia-twarzy/CelebA/Anno/identity_CelebA.txt', header=None, sep=" ")
ids = data[1].unique()
first_fifty = ids[:50]
images_path = "/home/adrian/Downloads/zdjęcia-twarzy/CelebA/Img/img_align_celeba"
image_temp = "/home/adrian/Downloads/zdjęcia-twarzy/CelebA/Img/temp"

for person_id in first_fifty:
    images_for_peron = data.loc[data[1] == person_id][0].array
    images_for_tests = [images_for_peron[-1]]
    images_for_training = images_for_peron[:-1]

    for file_name in images_for_peron:
        full_path = path.join(images_path, file_name)
        os.makedirs(
            path.join(image_temp, 'train', str(person_id)),
            exist_ok=True
        )

        copyfile(
            path.join(images_path, file_name),
            path.join(image_temp, 'train', str(person_id), file_name)
        )

    for file_name in images_for_tests:
        full_path = path.join(images_path, file_name)
        os.makedirs(
            path.join(image_temp, 'test', str(person_id), ),
            exist_ok=True
        )

        copyfile(
            path.join(images_path, file_name),
            path.join(image_temp, 'test', str(person_id), file_name)
        )
