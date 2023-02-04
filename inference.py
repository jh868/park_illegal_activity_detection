import torch
import os
import glob
import cv2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = torch.hub.load('ultralytics/yolov5', 'custom', path="./runs/train/exp_020212/weights/best.pt")
model.conf = 0.5  # NMS confidence threshold
model.iou = 0.45  # NMS IoU threshold
model.to(device)

image_dir = "./dataset/test/images/"
image_path = glob.glob(os.path.join(image_dir, "*.jpg"))

for img_path in image_path:
    img = cv2.imread(img_path)
    result = model(img, size=640)
    cv2.imshow('result', result.render()[0])
    cv2.waitKey(0)

