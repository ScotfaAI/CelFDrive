# Train the model in parralel using export MKL_SERVICE_FORCE_INTEL=1
from ultralytics import YOLO
from multiprocessing import freeze_support

def main():
    # Load the model from file
    model = YOLO('Models/Base/yolov9c.pt')
    dropout = 0.0
    device = [0]
    
    # Train the model
    results = model.train(
        imgsz=640,
        batch=16,
        epochs=150,
        patience=50,
        mosaic=0.0,
        flipud=0.5,
        dropout=dropout,
        data="Yaml/Scott40xonly.yaml",
        project="yolov9",
        name=f"20240513_Scott40xonly_{str(dropout)}_",
        device=device
    )

if __name__ == '__main__':
    freeze_support()
    main()
