from ultralytics import YOLOv10
model = YOLOv10("/root/autodl-tmp/yolov10/runs/detect/v8s(NEU)/weights/best.pt")
model.predict(source='/root/autodl-tmp/yolov10/val_test_img2',
              imgsz=640,
              project='runs/test',
              name='predict',
              save=True,
              conf=0.87,
              # visualize=True # visualize model features maps
              )

# save=True   '/root/autodl-tmp/yolov10/datasets/cocoRELyolo1/images/val2017'

