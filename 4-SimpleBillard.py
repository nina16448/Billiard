import numpy as np
import cv2
import json
from math import *
from time import *
from pathlib import Path
import time
import matplotlib.pyplot as plt
import numpy.polynomial.polynomial as poly
from random import random
import os
import torch
import copy

from models.common import DetectMultiBackend
from utils.datasets import letterbox
from utils.general import (
    check_img_size,
    non_max_suppression,
    apply_classifier,
    scale_coords,
    scale_boxes,
    xyxy2xywh,
    strip_optimizer,
    set_logging,
    increment_path,
)
from utils.cv_puttext import cv2ImgAddText
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, smart_inference_mode

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
############### Setting Importation ###############

with open("camera.json", "r") as f:
    data = json.load(f)

for k, v in data.items():
    globals()[k] = v

print("Camera Data:", data)

with open("data.json", "r") as f:
    data = json.load(f)

for k, v in data.items():
    globals()[k] = np.array(v)

print("Calibraton Data:", data)

############### Configuration Fenetre ###############
cv2.namedWindow("Billard", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty(
    "Billard", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN
)  # , cv2.WINDOW_NORMAL, cv2.WINDOW_NORMAL) #


cap = cv2.VideoCapture(camera_number)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH_MAX)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT_MAX)
cap.set(cv2.CAP_PROP_FPS, 30)
print("width: ", cap.get(cv2.CAP_PROP_FRAME_WIDTH))
print("height: ", cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print("fps: ", cap.get(cv2.CAP_PROP_FPS))


ret, frame = cap.read()
l, h, c = frame.shape
cadre = (WIDTH_MAX, HEIGHT_MAX)
print(l, h)


flagg = False
clean_time = -3
put_time = -5
POSlist = []
ipPOSlist = []


class Ball:
    scoreLimit = 0.90
    pointLimit = 15  # points
    lBall = []
    id_count = 0
    radius = 27
    mass = 111
    deceleration = 26
    memory = 10  # Le nombre de point utiliser pour les calcules
    max_ball = 20
    sId_available = set(range(max_ball))
    movestate = 1

    def __init__(self, t, x, y):
        global space, t_physic_engine
        Ball.lBall += [self]
        self.lPos = [[t, x, y]]
        self.lPos_prediction = [[t_physic_engine, x, y]]
        self.lVitesse = [[t, 0, 0, 0]]  # [[t, dx, dy, v]]
        self.lVitesse_prediction = [[t_physic_engine, 0, 0, 0]]
        self.polynome_distance = poly.Polynomial([0])
        self.debut_simulation = 0
        self.lChemin = []
        self.lVt1 = []
        self.lVt2 = []
        self.lTest = []
        self.lAngle = []
        self.lAcc = [[t, 0]]
        self.distance = 0
        self.std = []
        self.id = min(Ball.sId_available)
        Ball.sId_available.remove(self.id)
        self.explosion_frame = 0
        self.debut_simulation = time.time()
        self.movement = 4
        self.prediction_run = False
        self.creation_time = time.time()

    def __str__(self):
        return f"Ball ID: {self.id}"

    def __del__(self):
        global space
        Ball.sId_available.add(self.id)
        if self in Ball.lBall:
            Ball.lBall.remove(self)
        del self

    def interpolation(self, dt):
        tv, dx, dy, v = self.lVitesse[-1]
        t, x, y = self.lPos[-1]
        dp = self.polynome_distance(tv) - self.polynome_distance(tv + dt)
        px = x + dx * v * dt
        py = y + dy * v * dt
        px = min(max(px, 1), (WIDTH_MAX - 1))
        py = min(max(py, 1), (HEIGHT_MAX - 1))
        # x += dx*dp
        # y += dy*dp

        if v < 15:
            v = 0
            dx = 0
            dy = 0

        vx = dx * v
        vy = dy * v
        # print("!!!!!!!!")
        # print(dx, ", ", dy)
        return px, py, vx, vy, v

    def add_pos(self, t, x, y):
        self.movestate = 0
        global space, t_physic_engine
        # Le déplacement est-il suffisant ?
        if len(self.lPos) >= 1:
            _, x1, y1 = self.lPos[-1]
            d = ((x1 - x) ** 2 + (y1 - y) ** 2) ** 0.5
            if d < 10:
                _, x, y = self.lPos[-1]

        # Limite le nombre de donné pour notre prédiction
        aAverage = np.array(self.lPos[-Ball.memory + 1 :] + [[t, x, y]])

        self.lPos += [
            list(
                np.average(aAverage, weights=np.exp((aAverage[:, 0] - t) / 0.1), axis=0)
            )
        ]  # [[t, x, y], ...]

        if len(self.lPos) < ball.memory:
            return

        a = np.array(self.lPos[-2:])
        dt, dx, dy = a[1] - a[0]
        vx, vy = dx / dt, dy / dt

        self.lTest += [[t, (vx**2 + vy**2) ** 0.5]]

        lFitting_pos = self.lPos[-Ball.memory :]
        aFitting_pos = np.array(lFitting_pos)

        [[dx], [dy], [cx], [cy]] = cv2.fitLine(
            np.array([[[x, y]] for t, x, y in lFitting_pos]), cv2.DIST_L2, 0, 0.01, 0.01
        )

        # Déduit le sens:
        p0 = np.array(lFitting_pos[-2][1:3])
        p1 = np.array([cx + 10 * dx, cy + 10 * dy])
        p2 = np.array([cx - 10 * dx, cy - 10 * dy])

        d1 = np.linalg.norm(p0 - p1)
        d2 = np.linalg.norm(p0 - p2)

        if d2 < d1:
            dx *= -1
            dy *= -1

        # Projection sur l'axe de la direction:
        aFitting_pos = np.array(lFitting_pos)
        lT = aFitting_pos[:, 0]
        lXY = aFitting_pos[:, 1:3]
        rotation_matrix = np.array([[dx, -dy], [dy, dx]]).T
        lDistance = rotation_matrix.dot(lXY.T)[0, :]

        t_mean = (lT[-1] + lT[0]) / 2

        # Interpolation polynomiale de l'avancement:
        coefs = poly.polyfit(lT, lDistance, 2)

        pd = poly.Polynomial(coefs)
        pv = pd.deriv()
        pa = pv.deriv()

        self.polynome_distance = pd
        v = pv(t_mean)

        acc = pa(t_mean)

        # Limiter les vitesses trop petite:
        if v < 15:
            vx, vy, v = 0, 0, 0

        # Limiter les vitesses trop petite:
        if abs(acc) < 1:
            acc = 0

        self.lAngle += [[t, np.arctan2(dx, dy)]]

        aAverage = np.array(self.lVitesse[-Ball.memory + 1 :] + [[t_mean, dx, dy, v]])

        self.lVitesse += [
            list(
                np.average(
                    aAverage, weights=np.exp((aAverage[:, 0] - t) / 0.03), axis=0
                )
            )
        ]  # [[t, vx, vy, v], ...]

        aAverage = np.array(self.lAcc[-Ball.memory + 1 :] + [[t_mean, acc]])

        self.lAcc += [
            list(
                np.average(
                    aAverage, weights=np.exp((aAverage[:, 0] - t) / 1000), axis=0
                )
            )
        ]  # [[t, acc], ...]

        _, px, py = self.lPos[-(Ball.memory + 1) // 2]
        self.lChemin += [[px, py]]
        pv = self.lVitesse[-1][3]
        sV = 11
        self.lVt1 += [[px - dy * pv / sV, py + dx * pv / sV]]
        self.lVt2 += [[px + dy * pv / sV, py - dx * pv / sV]]

        if v > 20:
            if not self.prediction_run:
                self.movestate = 1
                print("Launch", t, self.prediction_run, "id", self.id)
                dt = (time.time() - t_prediction) * 7
                px = min(max(x + dx * v * dt, 1), (WIDTH_MAX - 1))
                py = min(max(y + dy * v * dt, 1), (HEIGHT_MAX - 1))

                scale = 100

                self.prediction_run = True
                self.debut_simulation = 0

        if v == 0:
            self.prediction_run = False

            self.lPos_prediction = [[t_physic_engine, x, y]]

            self.lChemin = []
            self.lVt1 = []
            self.lVt2 = []

    @classmethod
    def mapping_detecting_balls(cls, t, lDetected_ball):
        global flagg
        global POSlist
        global ipPOSlist
        global clean_time, put_time
        for ball in Ball.lBall:
            ball.movestate = 0
            if t - ball.lPos[-1][0] > 1.5:
                clean_time = t
                put_time = 0
                print("clean!")
                flagg = False
                POSlist = []
                ipPOSlist = []
                ball.__del__()

        if not lDetected_ball:
            return

        lDistance = []
        lMapped_ball = set()
        lMapped_detected_ball = set()

        for ball in Ball.lBall:
            _, x1, y1 = ball.lPos[-1]
            for x2, y2 in lDetected_ball:
                # assert not ((x2-x1)**2 + (y2-y1)**2)**0.5 in dDistance, "BUG !!!!!!!"
                lDistance += [
                    [((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5, (x2, y2), ball]
                ]

        lDistance = sorted(lDistance, key=lambda x: x[0])

        i = 0
        for d, detected, ball in lDistance:
            if i >= len(Ball.lBall):
                break

            if (not detected in lMapped_detected_ball) and (not ball in lMapped_ball):
                lMapped_detected_ball.add(detected)
                lMapped_ball.add(ball)
                lDetected_ball.remove(detected)
                ball.add_pos(t, *detected)
                i += 1

        for detected in lDetected_ball:
            Ball(t, *detected)


_, frame = cap.read()

prevframe = frame[:, :, :]  # frame[:,:,2]    #First frame
prevframe = cv2.warpPerspective(
    prevframe, m_camera2screen, (WIDTH_MAX, HEIGHT_MAX), flags=cv2.INTER_LINEAR
)
cv2.imshow("Billard", prevframe)

background = cv2.imread("background.jpg")[:, :, 2]
background = cv2.warpPerspective(
    background, m_camera2screen, (WIDTH_MAX, HEIGHT_MAX), flags=cv2.INTER_LINEAR
)

debut_time = time.time()
n_frame = 0
first = True
t_prediction = time.time()
t_physic_engine = 0
state = 0  # 0 靜止狀態 1 運動狀態

# for i in range(250):
#    ret, frame = cap.read()
rec_time = -5

maxx = 0
prev_M = []

while True:
    t_frame = time.time()
    ret, frame = cap.read()
    nextframe = frame[:, :, 2].copy()
    # frame = cv2.warpPerspective(
    #     frame, m_camera2screen, (WIDTH_MAX, HEIGHT_MAX), flags=cv2.INTER_LINEAR
    # )
    # background = frame[:, :, 2]
    # newframe = background
    nextframe = cv2.warpPerspective(
        nextframe, m_camera2screen, (WIDTH_MAX, HEIGHT_MAX), flags=cv2.INTER_LINEAR
    )

    nextframe = cv2.absdiff(background, nextframe)

    nextframe = cv2.GaussianBlur(nextframe, (5, 5), 0)

    _, nextframe = cv2.threshold(nextframe, 100, 255, cv2.THRESH_BINARY)
    # newframe = nextframe
    contours, hierarchy = cv2.findContours(
        nextframe, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )

    # newframe = fond.copy()  # np.zeros(frame.shape)
    newframe = cv2.warpPerspective(
        frame, m_camera2screen, (WIDTH_MAX, HEIGHT_MAX), flags=cv2.INTER_LINEAR
    )

    l = []

    for c in contours:
        M = cv2.moments(c)
        cv2.drawContours(newframe, contours, -1, (255, 0, 255), 2)

        area = cv2.contourArea(c)
        perimeter = cv2.arcLength(c, True)

        if perimeter == 0:
            continue

        circularity = 4 * np.pi * area / (perimeter * perimeter)

        # print("area", area)
        # print("perimeter", perimeter)
        # print("circularity", circularity)
        # Surface trop petite ?
        if M["m00"] < np.pi * 25**2:
            continue

        lX = [x for [[x, _]] in c]
        lY = [y for [[_, y]] in c]

        if np.corrcoef(lX, lY)[0, 1] ** 2 > 0.75:  # 球杆延伸線
            [vx, vy, x, y] = cv2.fitLine(c, cv2.DIST_L2, 0, 0.01, 0.01)
            cv2.line(
                newframe,
                tuple(map(int, (x + vx * -WIDTH_MAX, y + vy * -WIDTH_MAX))),
                tuple(map(int, (x + vx * WIDTH_MAX, y + vy * WIDTH_MAX))),
                (255, 255, 255),
                15,
            )
            # p = np.polyfit(lX, lY, 1)
            # cv2.line(nextframe, (0, int(np.polyval(p, 0))), (WIDTH_MAX, int(np.polyval(p, WIDTH_MAX))), 150, 20)
            continue
        # if maxx < circularity:
        #     maxx = circularity
        #     print(maxx)
        # print("circularity", circularity)
        if circularity < 0.55:
            continue

        x = int(M["m10"] / M["m00"])
        y = int(M["m01"] / M["m00"])
        ecartype = np.std(
            [((x - ix) ** 2 + (y - iy) ** 2) ** 0.5 for ix, iy in zip(lX, lY)]
        )

        # 它看起來像一個圓圈嗎？
        # if ecartype < 10:
        # Ball.add_ball(t_frame, x, y)
        # cv2.circle(newframe, (x, y), 50, (255, 255, 255), 10)
        l += [(x, y)]

        prev_M = M

    Ball.mapping_detecting_balls(t_frame - debut_time, l)

    for ball in Ball.lBall:
        rx, ry, vx, vy, v = ball.interpolation((time.time() - t_prediction) + 0.4)
        t, x, y = ball.lPos[-1]
        atv, adx, ady, av = ball.lVitesse[-1]
        rx, ry = int(rx), int(ry)
        x, y = int(x), int(y)
        zoom = 0.7
        aPos_x = (3.5 - float(y / float(HEIGHT_MAX) * 7)) * 1
        aPos_y = 5.548624
        aPos_z = (7.2 - float(x / float(WIDTH_MAX) * 14.4)) * 1

        if flagg == False and v == 0:
            if put_time == 0:
                put_time = t
            if t - put_time > 5:
                POSlist = []
                print("New Position: ", ball)
                print(aPos_x, ", ", aPos_y, ", ", aPos_z)
                Posdata = {
                    "Position": {"x": aPos_x, "y": aPos_y, "z": aPos_z},
                }
                with open("Position.json", "w") as f:
                    json.dump(Posdata, f, indent=4)  # 使用indent參數來讓輸出的json格式有縮排，看起來更整潔
                flagg = True

        if ball.movestate == 1 and v > 0 and t - rec_time > 10 and flagg == True:
            # 上傳資料 只剩下範圍問題:))
            # flagg = False
            rec_time = t
            print("Hit")
            HitSpeed = v  # 0-8
            if HitSpeed > 700:
                HitSpeed = 700
            # HitSpeed = float(HitSpeed / 700) * 8.0
            HitSpeed = np.log2(HitSpeed) - 2.5
            Dir_x = -ady  # 1 - -1
            Dir_y = 0
            Dir_z = -adx
            print("Hit: ", ball)
            print(HitSpeed)
            print(Dir_x, ", ", Dir_y, ", ", Dir_z)
            # 建立一個字典來存放你的變數
            Hitdata = {
                "HitSpeed": HitSpeed,
                "HitDirection": {"x": Dir_x, "y": Dir_y, "z": Dir_z},
            }
            with open("Hit.json", "w") as f:
                json.dump(Hitdata, f, indent=4)  # 使用indent參數來讓輸出的json格式有縮排，看起來更整潔

        # print("state", state)

        # for road in ipPOSlist:
        #     cv2.circle(newframe, road, 50, (0, 0, 0), 10)

        for road in POSlist:
            cv2.circle(newframe, road, 50, (255, 0, 0), 10)

        cv2.circle(newframe, (x, y), 50, (255, 255, 255), 10)

        if (x, y) not in POSlist:
            POSlist.append((x, y))

        cv2.circle(
            newframe,
            (x, y),
            int(Ball.radius),
            (0, 255, 0),
            -1,
        )

        font_scale = cv2.getFontScaleFromHeight(cv2.FONT_HERSHEY_SIMPLEX, 30, 3)
        cv2.putText(
            newframe,
            "ID:" + chr(ord("A") + ball.id),
            (int(x + Ball.radius * 2.2), y + 33),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            newframe,
            "V=" + str(round(v)),
            (int(x + Ball.radius * 2.2), y - 3),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    cv2.putText(
        newframe,
        "FPS: " + str(int(1 / (time.time() - t_frame))),
        (50, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        2,
        (255, 255, 255),
        3,
        cv2.LINE_AA,
    )

    step = 1 / 120
    n = int((time.time() - debut_time - t_physic_engine) / step)

    cv2.imshow("Billard", newframe)

    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break
    elif k == ord("r"):
        POSlist = []

cv2.destroyAllWindows()
cap.release()
exit()
