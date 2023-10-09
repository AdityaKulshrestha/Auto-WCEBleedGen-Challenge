from ultralytics import YOLO


def train_model():
    model = YOLO('yolov8n.pt')
    results = model.train(data='segmentation/config.yaml', epochs=5, imgsz=224)
    return results


if __name__ == '__main__':
    results = train_model()
    print(results)
