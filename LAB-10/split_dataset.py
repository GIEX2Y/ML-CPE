import os
import random
import shutil


SOURCE = "original_dataset"

IMAGE_PATH = SOURCE + "/images"
LABEL_PATH = SOURCE + "/labels"



sets = {
    "train":400,
    "val":50,
    "test":50
}


for folder in sets:

    os.makedirs(
        f"dataset/images/{folder}",
        exist_ok=True
    )

    os.makedirs(
        f"dataset/labels/{folder}",
        exist_ok=True
    )


images = os.listdir(IMAGE_PATH)

random.shuffle(images)


start = 0


for folder, amount in sets.items():

    selected = images[start:start+amount]

    start += amount


    for img in selected:

        name = img.replace(".jpg","")


        shutil.copy(
            f"{IMAGE_PATH}/{img}",
            f"dataset/images/{folder}/{img}"
        )


        shutil.copy(
            f"{LABEL_PATH}/{name}.txt",
            f"dataset/labels/{folder}/{name}.txt"
        )


print("Dataset Split Complete")