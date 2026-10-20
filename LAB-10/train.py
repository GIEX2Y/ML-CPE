from ultralytics import YOLO



configs = [
    ("YOLO_10_epoch",10),
    ("YOLO_30_epoch",30),
    ("YOLO_50_epoch",50)
]



for name, epoch in configs:


    print(
        "Training",
        name
    )


    model = YOLO(
        "yolov8n.pt"
    )


    model.train(

        data="data.yaml",

        epochs=epoch,

        imgsz=416,

        batch=8,

        name=name
    )



print("Training Finished")