# from IPython import display
# display.clear_output()

# import ultralytics
# ultralytics.checks()
# import os
# HOME = os.getcwd()
# print(HOME)

# from ultralytics import YOLO

# from IPython.display import display, Image

# from roboflow import Roboflow
# # rf = Roboflow(api_key="3OEqKlaUTLNgrFZGJiv6")
# # project = rf.workspace("kashish").project("3riders")
# # version = project.version(2)
# # dataset = version.download("yolov8")


# !yolo task=detect mode=train model=yolov8s.pt data="D:/TestingProjects/TrafficViolations/data.yaml" epochs=10 imgsz=800 plots=True


# from inference_sdk import InferenceHTTPClient

# CLIENT = InferenceHTTPClient(
#     api_url="https://serverless.roboflow.com",
#     api_key="3OEqKlaUTLNgrFZGJiv6"
# )

# result = CLIENT.infer("1.jpg", model_id="3riders/2")



# from inference_sdk import InferenceHTTPClient
# from PIL import Image
# import requests
# from io import BytesIO

# # Initialize Roboflow client
# CLIENT = InferenceHTTPClient(
#     api_url="https://serverless.roboflow.com",
#     api_key="3OEqKlaUTLNgrFZGJiv6"
# )

# # Run inference on the image (must be local path)
# result = CLIENT.infer("1.jpg", model_id="3riders/2")

# # Optional: Print prediction results
# print(result)

# # Get image with prediction overlays from Roboflow response
# if "image" in result:
#     image_url = result["image"]  # URL of image with overlays
#     response = requests.get(image_url)

#     # Display the image using PIL
#     img = Image.open(BytesIO(response.content))
#     img.show()
# else:
#     print("No image URL in result.")


# from roboflow import Roboflow
# import supervision as sv
# import cv2

# rf = Roboflow(api_key="3OEqKlaUTLNgrFZGJiv6")
# project = rf.workspace().project("3riders")
# model = project.version(2).model

# result = model.predict("1.jpg", confidence=40, overlap=30).json()

# labels = [item["class"] for item in result["predictions"]]

# detections = sv.Detections.from_roboflow(result)

# label_annotator = sv.LabelAnnotator()
# bounding_box_annotator = sv.BoxAnnotator()

# image = cv2.imread("your_image.jpg")

# annotated_image = box_annotator.annotate(
#     scene=image, detections=detections)
# annotated_image = label_annotator.annotate(
#     scene=annotated_image, detections=detections, labels=labels)

# sv.plot_image(image=annotated_image, size=(16, 16))

# from roboflow import Roboflow
# import supervision as sv
# import cv2
# import numpy as np

# # Load image path
# image_path = "1.jpg"

# # Initialize Roboflow
# rf = Roboflow(api_key="3OEqKlaUTLNgrFZGJiv6")
# project = rf.workspace().project("3riders")
# model = project.version(2).model

# # Make prediction
# result = model.predict(image_path, confidence=40, overlap=30).json()

# # Parse predictions
# predictions = result["predictions"]

# # Prepare data for Detections object
# xyxy = []
# confidences = []
# class_ids = []
# labels = []

# for pred in predictions:
#     x1 = int(pred["x"] - pred["width"] / 2)
#     y1 = int(pred["y"] - pred["height"] / 2)
#     x2 = int(pred["x"] + pred["width"] / 2)
#     y2 = int(pred["y"] + pred["height"] / 2)

#     xyxy.append([x1, y1, x2, y2])
#     confidences.append(pred["confidence"])
#     class_ids.append(pred.get("class_id", 0))
#     labels.append(pred["class"])

# # Convert to NumPy arrays
# detections = sv.Detections(
#     xyxy=np.array(xyxy),
#     confidence=np.array(confidences),
#     class_id=np.array(class_ids)
# )

# # Load original image
# image = cv2.imread(image_path)

# # Annotate
# box_annotator = sv.BoxAnnotator()
# label_annotator = sv.LabelAnnotator()

# annotated_image = box_annotator.annotate(scene=image.copy(), detections=detections)
# annotated_image = label_annotator.annotate(scene=annotated_image, detections=detections, labels=labels)

# # Display image using OpenCV (you can use matplotlib if you prefer)
# cv2.imshow("Annotated Image", annotated_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# # Save annotated image
# cv2.imwrite("output.jpg", annotated_image)
# print("✅ Annotated image saved as output.jpg")


# from roboflow import Roboflow
# import supervision as sv
# import cv2
# import numpy as np
# import matplotlib.pyplot as plt

# # Load image path
# image_path = "1.jpg"

# # Initialize Roboflow
# rf = Roboflow(api_key="3OEqKlaUTLNgrFZGJiv6")
# project = rf.workspace().project("3riders")
# model = project.version(2).model

# # Make prediction
# result = model.predict(image_path, confidence=40, overlap=30).json()

# # Parse predictions
# predictions = result["predictions"]

# # Prepare data for Detections object
# xyxy = []
# confidences = []
# class_ids = []
# labels = []

# for pred in predictions:
#     x1 = int(pred["x"] - pred["width"] / 2)
#     y1 = int(pred["y"] - pred["height"] / 2)
#     x2 = int(pred["x"] + pred["width"] / 2)
#     y2 = int(pred["y"] + pred["height"] / 2)

#     xyxy.append([x1, y1, x2, y2])
#     confidences.append(pred["confidence"])
#     class_ids.append(pred.get("class_id", 0))
#     labels.append(pred["class"])

# # Convert to NumPy arrays
# detections = sv.Detections(
#     xyxy=np.array(xyxy),
#     confidence=np.array(confidences),
#     class_id=np.array(class_ids)
# )

# # Load original image
# image = cv2.imread(image_path)
# image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# # Annotate
# box_annotator = sv.BoxAnnotator()
# label_annotator = sv.LabelAnnotator()

# annotated_image = box_annotator.annotate(scene=image_rgb.copy(), detections=detections)
# annotated_image = label_annotator.annotate(scene=annotated_image, detections=detections, labels=labels)

# # Display using matplotlib
# plt.figure(figsize=(10, 10))
# plt.imshow(annotated_image)
# plt.axis('off')
# plt.title("Annotated Image")
# plt.show()

# # Save annotated image
# cv2.imwrite("output.jpg", cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))
# print("✅ Annotated image saved as output.jpg")


