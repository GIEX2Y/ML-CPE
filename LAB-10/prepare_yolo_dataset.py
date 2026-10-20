import os
import shutil
import random


SOURCE = "images"

OUTPUT = "dataset"


# จำนวนรูป
TOTAL = 500


# สร้าง Folder
folders = [
    "images/train",
    "images/val",
    "images/test",
    "labels/train",
    "labels/val",
    "labels/test"
]


for folder in folders:
    os.makedirs(
        OUTPUT + "/" + folder,
        exist_ok=True
    )


# อ่านรูป Cat Dog
files = []


for cls, folder in enumerate(["Cat", "Dog"]):

    path = os.path.join(SOURCE, folder)

    for img in os.listdir(path):

        if img.lower().endswith(
            (".jpg",".jpeg",".png")
        ):

            files.append(
                (
                    os.path.join(path,img),
                    cls
                )
            )


# สุ่ม 500 รูป
random.shuffle(files)

files = files[:TOTAL]


# แบ่งข้อมูล
train = files[:400]
val = files[400:450]
test = files[450:500]


def create_data(data, split):

    for index,(img_path,cls) in enumerate(data):

        name = f"{split}_{index}.jpg"


        # copy image
        shutil.copy(
            img_path,
            f"{OUTPUT}/images/{split}/{name}"
        )


        # สร้าง YOLO label
        label_name = name.replace(
            ".jpg",
            ".txt"
        )


        with open(
            f"{OUTPUT}/labels/{split}/{label_name}",
            "w"
        ) as f:

            # box เต็มภาพ
            f.write(
                f"{cls} 0.5 0.5 1.0 1.0"
            )


create_data(train,"train")
create_data(val,"val")
create_data(test,"test")


print("Complete")
print("Train:",len(train))
print("Val:",len(val))
print("Test:",len(test))