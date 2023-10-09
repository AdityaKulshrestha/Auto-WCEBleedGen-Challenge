from ultralytics import YOLO
import argparse


def train_model(config_path: str = 'segmentation/config.yaml', epochs: int = 5, imgsz: int = 224):
    model = YOLO('yolov8n.pt')
    results = model.train(data=config_path, epochs=epochs, imgsz=imgsz)
    return results


parser = argparse.ArgumentParser(description="Training Script for YOLO")
parser.add_argument('--config_path')
parser.add_argument('--epochs')
parser.add_argument('--imgsz')

args = parser.parse_args()
config_path = args.config_path
epochs = args.epochs
image_size = args.imgsz

results = train_model(config_path, epochs, image_size)
print(results)
