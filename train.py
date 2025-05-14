import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('/root/autodl-tmp/yolov10/yolov10s.yaml') # /root/ultralytics-main/yolov8s.yaml

    model.train(**{'cfg': '/root/autodl-tmp/yolov10/ultralytics/cfg/default.yaml', 'data': '/root/autodl-tmp/yolov10/datasets/coco.yaml'})

# 断点续训
# from ultralytics.models.yolov10.model import YOLOv10
# model = YOLOv10(r"/root/autodl-tmp/yolov10/runs/detect/train8/weights/last.pt") #替换成上一次中断时的权重文件路径
# model.train(resume=True) #保持上一次的参数设置使用resume=True，也可以使用其他训练参数重新训练
