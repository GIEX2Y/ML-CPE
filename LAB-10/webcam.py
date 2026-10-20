from ultralytics import YOLO
import cv2


model = YOLO(
    "runs/detect/YOLO_catdog_50epoch/weights/best.pt"
)


cap = cv2.VideoCapture(0)


while True:

    ret, frame = cap.read()

    if not ret:
        break


    results = model(frame)


    frame = results[0].plot()


    cv2.imshow(
        "YOLO Real Time Detection",
        frame
    )


    if cv2.waitKey(1) == ord("q"):
        break


cap.release()
cv2.destroyAllWindows()