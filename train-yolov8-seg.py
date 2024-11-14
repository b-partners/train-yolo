from ultralytics import YOLO

# Load a model
model = YOLO("yolov8x-seg.pt")

# Train the model
train_results = model.train(
    data="data.yaml",  # path to dataset YAML
    epochs=100,  # number of training epochs
    imgsz=1024,  # training image size
    device=[0,1,2,3]  # device to run on, i.e. device=0 or device=0,1,2,3 or device=cpu
)

#model.data_check("data.yaml")

# Load a model
model = YOLO("runs/segment/train3/weights/best.pt")

# Perform object detection on an image
results = model("test_images", save=True)