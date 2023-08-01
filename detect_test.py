import time
from pathlib import Path
import torch
import numpy as np
import json
import copy
import os

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

# from models.experimental import attempt_load
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
import cv2


############### Setting Importation ###############
WIDTH_MAX = 1280
HEIGHT_MAX = 960
clors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255)]

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

###### Yolo初始化 ############
device = ""
device = select_device(device)
# detect_model = attempt_load("./cue_cueball.pt", map_location=device)  # load FP32 model

detect_model = DetectMultiBackend(
    "./cue_cueball.pt", device=device, dnn=False, data="./data/coco128.yaml", fp16=False
)
stride, names, pt = detect_model.stride, detect_model.names, detect_model.pt


def scale_coords_landmarks(img1_shape, coords, img0_shape, ratio_pad=None):  # 返回到原图坐标
    # Rescale coords (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(
            img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1]
        )  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (
            img1_shape[0] - img0_shape[0] * gain
        ) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    print("coords shape:", coords.shape)
    print("pad[0]:", pad[0])

    coords[:, [0, 2, 4, 6]] -= pad[0]  # x padding
    coords[:, [1, 3, 5, 7]] -= pad[1]  # y padding
    coords[:, :8] /= gain
    # clip_coords(coords, img0_shape)
    coords[:, 0].clamp_(0, img0_shape[1])  # x1
    coords[:, 1].clamp_(0, img0_shape[0])  # y1
    coords[:, 2].clamp_(0, img0_shape[1])  # x2
    coords[:, 3].clamp_(0, img0_shape[0])  # y2
    coords[:, 4].clamp_(0, img0_shape[1])  # x3
    coords[:, 5].clamp_(0, img0_shape[0])  # y3
    coords[:, 6].clamp_(0, img0_shape[1])  # x4
    coords[:, 7].clamp_(0, img0_shape[0])  # y4
    # coords[:, 8].clamp_(0, img0_shape[1])  # x5
    # coords[:, 9].clamp_(0, img0_shape[0])  # y5
    return coords


def detect(model, img):
    conf_thres = 0.45
    iou_thres = 0.45
    img_size = 640
    dict_list = []
    img0 = copy.deepcopy(img)
    h0, w0 = img.shape[:2]
    r = img_size / max(h0, w0)
    if r != 1:  # 縮放圖片到640*640
        interp = cv2.INTER_AREA if r < 1 else cv2.INTER_LINEAR
        img0 = cv2.resize(img0, (int(w0 * r), int(h0 * r)), interpolation=interp)

    imgsz = check_img_size(img_size, s=model.stride)  # check img_size

    img0 = letterbox(img0, new_shape=imgsz)[0]  # 檢測前處理，圖片長寬變為32倍數
    img0 = img0[:, :, ::-1].transpose(2, 0, 1).copy()
    # ^BGR to RGB, to 3x416x416  图片的BGR排列转为RGB,然后将图片的H,W,C排列变为C,H,W排列

    # Run inference
    t0 = time.time()

    img0 = torch.from_numpy(img0).to(device)
    img0 = img0.float()  # uint8 to fp16/32
    img0 /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img0.ndimension() == 3:
        img0 = img0.unsqueeze(0)

    pred = model(img0)[0]

    pred = non_max_suppression(pred, conf_thres, iou_thres)

    for i, det in enumerate(pred):  # detections per image
        gn = torch.tensor(img.shape)[[1, 0, 1, 0]]  # normalization gain whwh
        annotator = Annotator(img, line_width=3, example=str(names))
        if len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_boxes(img0.shape[2:], det[:, :4], img.shape).round()

            # Write results
            for *xyxy, conf, cls in reversed(det):
                xywh = (
                    (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()
                )  # normalized xywh
                line = (cls, *xywh)  # label format

                c = int(cls)  # integer class
                label = f"{names[c]} {conf:.2f}"
                annotator.box_label(xyxy, label, color=colors(c, True))

            # # Rescale boxes from img_size to im0 size
            # det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img.shape).round()

            # # Print results
            # for c in det[:, -1].unique():
            #     n = (det[:, -1] == c).sum()  # detections per class
            # # print("det[:, 5:13]:", det[:, 5:13])

            # # det[:, 5:13] = scale_coords_landmarks(
            # #     img.shape[2:], det[:, 5:13], img.shape
            # # ).round()
            # det[:, :4] = scale_boxes(img.shape[2:], det[:, :4], img.shape).round()

            # for j in range(det.size()[0]):
            #     result_dict = {}
            #     xyxy = det[j, :4].view(-1).tolist()
            #     x1 = int(xyxy[0])
            #     y1 = int(xyxy[1])
            #     x2 = int(xyxy[2])
            #     y2 = int(xyxy[3])
            #     rect = [x1, y1, x2, y2]

            #     conf = det[j, 4].cpu().numpy()
            #     # landmarks = det[j, 5:13].view(-1).tolist()
            #     # landmarks_np = np.zeros((4, 2))
            #     # for i in range(4):
            #     #     point_x = int(landmarks[2 * i])
            #     #     point_y = int(landmarks[2 * i + 1])
            #     # landmarks_np[i] = np.array([point_x, point_y])

            #     class_num = det[j, 6].cpu().numpy()
            #     class_label = int(class_num)

            #     result_dict["rect"] = rect
            #     result_dict["detect_conf"] = conf
            #     # result_dict["landmarks"] = landmarks_np.tolist()
            #     result_dict["plate_type"] = class_label

            #     dict_list.append(result_dict)

    return annotator.result()


while True:
    ret, frame = cap.read()
    l, h, c = frame.shape
    # print(l, h)

    frame = cv2.warpPerspective(frame, m_camera2screen, (h, l), flags=cv2.INTER_LINEAR)

    ori_img = detect(detect_model, frame)
    # ori_img = draw_result(frame, dict_list)
    cv2.imshow("Billard", ori_img)

    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break


cap.release()
cv2.destroyAllWindows()
