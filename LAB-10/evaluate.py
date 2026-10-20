from ultralytics import YOLO



models = [

"runs/detect/YOLO_10_epoch/weights/best.pt",

"runs/detect/YOLO_30_epoch/weights/best.pt",

"../mv ~/Desktop/ML-CPE/runs ~/Desktop/ML-CPE/LAB-10/runs/detect/YOLO_50_epoch/weights/best.pt"

]



for model_path in models:


    print("\nModel:",model_path)


    model = YOLO(model_path)


    result = model.val()



    print(
        "Precision:",
        result.box.mp
    )


    print(
        "Recall:",
        result.box.mr
    )


    print(
        "mAP50:",
        result.box.map50
    )


    print(
        "mAP50-95:",
        result.box.map
    )