import warnings
warnings.filterwarnings('ignore')
import numpy as np

from ultralytics import YOLOv10

if __name__ == '__main__':
    model = YOLOv10('/root/autodl-tmp/yolov10/runs/detect/Yolov3-tiny_NEU/weights/best.pt')
    model.val(data='/root/autodl-tmp/yolov10/datasets/coco.yaml',
              split='val',
              imgsz=640,
              batch=16,
              rect=False,
              save_json=True, # if you need to cal coco metrice
              project='runs/val',
              name='exp',
              )
