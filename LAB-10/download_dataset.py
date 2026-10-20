from datasets import load_dataset
import os
import shutil


# โหลด Dataset
ds = load_dataset(
    "bohan666/cat-dog-yolo-dataset"
)


print(ds)
print(ds["train"].column_names)


# สร้าง folder
folders = [
    "dataset/images/train",
    "dataset/images/val",
    "dataset/images/test",
    "dataset/labels/train",
    "dataset/labels/val",
    "dataset/labels/test"
]


for folder in folders:
    os.makedirs(folder, exist_ok=True)


# ดูจำนวนข้อมูล
data = ds["train"]

print("Total data:", len(data))


# ใช้ 500 รูป
data = data.select(range(500))


# แบ่งจำนวน
train_data = data.select(range(400))
val_data = data.select(range(400,450))
test_data = data.select(range(450,500))


def export_dataset(dataset, split):

    for i, item in enumerate(dataset):

        print(
            f"{split}: {i+1}/{len(dataset)}"
        )


        # ตรวจชื่อ field
        print(item.keys())

        break



export_dataset(train_data,"train")