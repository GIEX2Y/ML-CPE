from ultralytics import YOLO



model = YOLO(

"runs/detect/YOLO_50_epoch/weights/best.pt"

)



model.predict(

source="dataset/images/test",

conf=0.5,

save=True

)



print("Prediction complete")