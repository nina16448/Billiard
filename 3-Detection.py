import numpy as np
import cv2
from itertools import permutations
import json
import os

# WIDTH_MAX = 1920
# HEIGHT_MAX = 1080

############### Setting Importation ###############

with open("camera.json", "r") as f:
    data = json.load(f)

for k, v in data.items():
    globals()[k] = v

print("Camera Data:", data)

cv2.namedWindow("Billard", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("Billard", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
cap = cv2.VideoCapture(camera_number)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH_MAX)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT_MAX)

if os.path.isfile("debug.mp4"):
    cap = cv2.VideoCapture("debug.mp4")


while True:
    ok, frame = cap.read()
    frame2 = frame[:, :, 2].copy()
    cv2.putText(
        frame,
        "Clear the billard table, and when it's done, press any touch !",
        (30, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1 / 2,
        (0, 0, 255),
        1,
        cv2.LINE_AA,
    )

    if cv2.waitKey(30) != -1:
        break

    cv2.imshow("Billard", frame)

# frame2 = cv2.resize(frame2,(WIDTH_MAX,HEIGHT_MAX))

cv2.imwrite("background.jpg", frame2)

cv2.destroyAllWindows()
cap.release()
input("Detection finished")
exit()
