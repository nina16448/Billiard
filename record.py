import numpy as np
import cv2
import json

############### Trouver Caméra ###############
WIDTH_MAX = 1920
HEIGHT_MAX = 1080

cv2.namedWindow("Camera", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("Camera", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)


camera_number = "/dev/video3"


cap = cv2.VideoCapture(camera_number)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH_MAX)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT_MAX)
cap.set(cv2.CAP_PROP_FPS, 30)
print("width: ", cap.get(cv2.CAP_PROP_FRAME_WIDTH))
print("height: ", cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print("fps: ", cap.get(cv2.CAP_PROP_FPS))

fps = cap.get(cv2.CAP_PROP_FPS)
width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(
    cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
)
fourcc = cv2.VideoWriter_fourcc(*"XVID")
out = cv2.VideoWriter("result2.avi", fourcc, fps, (width, height))


while True:
    ok, frame = cap.read()

    cv2.imshow("Camera", frame)
    out.write(frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
cv2.destroyAllWindows()
cap.release()
exit()
