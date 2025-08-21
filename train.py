from ultralytics import YOLO

model = YOLO("yolov8m.pt")  

results = model.train(
    data="/kaggle/input/underwater-plastic-pollution-detection/underwater_plastics/data.yaml",
    epochs=120,        
    batch=16,          
    imgsz=640,
    device="0",
    name="yolov8m_plastic_v1",
    patience=20,       
    lr0=0.01,          
    hsv_h=0.1,         
    hsv_s=0.7,         
)
