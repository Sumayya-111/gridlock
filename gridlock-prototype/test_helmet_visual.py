from ultralytics import YOLO

model = YOLO("api/best.pt")

model.predict(
    source="api/testsample.mp4",
    conf=0.20,
    imgsz=640,
    save=True,
    project="runs/helmet_test",
    name="new_model",
    exist_ok=True,
    verbose=False
)

print("Test complete. Check the runs/helmet_test/new_model folder.")
