import numpy as np
import cv2
import json

############### Trouver Caméra ###############
WIDTH_MAX = 1920
HEIGHT_MAX = 1080
# WIDTH_MAX = 1280
# HEIGHT_MAX = 960
cv2.namedWindow("Camera", cv2.WND_PROP_FULLSCREEN)
# cv2.setWindowProperty("Camera", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)


camera_number = "/dev/video1"

while True:
    cap = cv2.VideoCapture(camera_number)
    cap.set(cv2.CAP_PROP_FPS, 30)
    print("fps: ", cap.get(cv2.CAP_PROP_FPS))
    test, frame = cap.read()

    if not test:
        assert camera_number != 0, "No Camera Available !!!"
        cap = cv2.VideoCapture(camera_number)
        test, frame = cap.read()

    frame = cv2.resize(frame, (WIDTH_MAX, HEIGHT_MAX))

    while True:
        ok, frame = cap.read()
        assert ok, "Camera disconnected"
        frame = cv2.resize(frame, (WIDTH_MAX, HEIGHT_MAX))

        cv2.putText(
            frame,
            "Press enter to valide, or press any other touch to switch of camera :",
            (30, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1 / 2,
            (0, 0, 255),
            1,
            cv2.LINE_AA,
        )

        cv2.imshow("Camera", frame)

        k = cv2.waitKey(1)
        if k != -1:
            break

    if k == 13:
        break

    camera_number += 1

cv2.destroyAllWindows()
cap.release()

data = {
    "camera_number": camera_number,
    "WIDTH_MAX": WIDTH_MAX,
    "HEIGHT_MAX": HEIGHT_MAX,
}
with open("camera.json", "w") as f:
    json.dump(data, f)

input("Calibration finished !!")
exit()
