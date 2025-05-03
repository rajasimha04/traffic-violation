
from roboflow import Roboflow
import supervision as sv
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load image path
image_path = "3.jpg"

# Initialize Roboflow

rf = Roboflow(api_key="3OEqKlaUTLNgrFZGJiv6")
project = rf.workspace().project("3riders")
model = project.version(2).model

# Make prediction
result = model.predict(image_path, confidence=40, overlap=30).json()

# Parse predictions
predictions = result["predictions"]
# print(f"Predictions: {predictions}")
# print(f"Number of predictions: {len(predictions)}")
# print(f"Prediction details: {predictions[0]}")  # Print details of the first prediction
# print(f"Prediction class: {predictions[0]['class']}")   # Print class of the first prediction   

unique_classes = set(pred["class"] for pred in predictions)
print("✅ Unique Detected Classes:")
for cls in unique_classes:
    print(f"- {cls}")



# Prepare data for Detections object
xyxy = []
confidences = []
class_ids = []
labels = []

for pred in predictions:
    x1 = int(pred["x"] - pred["width"] / 2)
    y1 = int(pred["y"] - pred["height"] / 2)
    x2 = int(pred["x"] + pred["width"] / 2)
    y2 = int(pred["y"] + pred["height"] / 2)

    xyxy.append([x1, y1, x2, y2])
    confidences.append(pred["confidence"])
    class_ids.append(pred.get("class_id", 0))
    labels.append(pred["class"])

# Convert to NumPy arrays
detections = sv.Detections(
    xyxy=np.array(xyxy),
    confidence=np.array(confidences),
    class_id=np.array(class_ids)
)

# Load original image
image = cv2.imread(image_path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Annotate
box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()

annotated_image = box_annotator.annotate(scene=image_rgb.copy(), detections=detections)
annotated_image = label_annotator.annotate(scene=annotated_image, detections=detections, labels=labels)

# Display using matplotlib
plt.figure(figsize=(10, 10))
plt.imshow(annotated_image)
plt.axis('off')
plt.title("Annotated Image")
plt.show()

# Save annotated image
cv2.imwrite("output.jpg", cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))
print("✅ Annotated image saved as output.jpg")
