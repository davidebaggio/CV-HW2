from ultralytics import YOLO

model = YOLO('yolov8n.pt')

model.train(data='cards.yaml', epochs=25, imgsz=640, batch=16, name='card_detector', device='0')

model.export(format='onnx')