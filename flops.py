from ultralytics import YOLOv10

model = YOLOv10('/root/autodl-tmp/yolov10/yolov10s-bifpn.yaml')
model.model.model[-1].export = True
model.model.model[-1].format = 'onnx'
del model.model.model[-1].cv2
del model.model.model[-1].cv3
model.fuse()