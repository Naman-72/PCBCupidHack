from roboflow import Roboflow
import supervision as sv
import cv2

rf = Roboflow(api_key="iHkN8n2LTPw6hJuo7YiF")
project = rf.workspace().project("component-detection-e5wj9")
model = project.version(7).model

image_path = "yolo_test.jpg"

result = model.predict(image_path, confidence=40, overlap=30).json()

labels = [item["class"] for item in result["predictions"]]

detections = sv.Detections.from_inference(result)

label_annotator = sv.LabelAnnotator()
bounding_box_annotator = sv.BoxAnnotator()

image = cv2.imread(image_path)

if image is None:
    raise FileNotFoundError(f"Could not read image: {image_path}")

annotated_image = bounding_box_annotator.annotate(
    scene=image,
    detections=detections
)

annotated_image = label_annotator.annotate(
    scene=annotated_image,
    detections=detections,
    labels=labels
)

sv.plot_image(image=annotated_image, size=(16, 16))