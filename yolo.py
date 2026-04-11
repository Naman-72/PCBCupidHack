# from ultralytics import YOLO

# # Load the model
# model = YOLO("yolo26m.pt")

# # Run inference
# # results = model("yolo_test.jpg")
# # results[0].show()


# from inference_sdk import InferenceHTTPClient

# CLIENT = InferenceHTTPClient(
#     api_url="https://serverless.roboflow.com",
#     api_key="iHkN8n2LTPw6hJuo7YiF"
# )

# result = CLIENT.infer("yolo_test.jpg", model_id="component-detection-e5wj9/7")


from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from sahi.utils.cv import read_image_as_pil
