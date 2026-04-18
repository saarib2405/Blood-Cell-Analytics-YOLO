from ultralytics import YOLO
from pathlib import Path
import torch

def main():
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Device Name: {torch.cuda.get_device_name(0)}")

    # Load a pre-trained model (recommended for training)
    # Using the nano model as it trains faster and is good for getting started
    model = YOLO("yolo11n.pt")
    
    # Get absolute path to data.yaml
    data_path = Path(__file__).parent / "data.yaml"

    # Train the model
    results = model.train(
        data=str(data_path.absolute()),
        epochs=30,          # Reduced epochs for quicker turnaround since we are developing
        imgsz=640,
        device=0 if torch.cuda.is_available() else 'cpu',
        batch=16,
        name="blood_cell_model",
        exist_ok=True       # Overwrite existing if run multiple times
    )

    print("Training completed. Model saved at runs/detect/blood_cell_model/weights/best.pt")

if __name__ == "__main__":
    main()
