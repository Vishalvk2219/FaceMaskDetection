import cv2
import torch
import torchvision.transforms as transforms
import mediapipe as mp
import numpy as np
from torchvision import models
import torch.nn as nn

# Load model

model = models.mobilenet_v3_large(pretrained=True)
num_ftrs = model.classifier[3].in_features  
model.classifier[3] = nn.Linear(num_ftrs, 3)
model.load_state_dict(torch.load("mobilenetv3_mask_model.pth", map_location=torch.device('cpu')))
model.eval()


# Classes and colors
classes = ['Mask', 'No Mask','Mask_Weared_Incorrect' ]
colors = [(0, 255, 0), (0, 0, 255), (255,0,0) ]

# Transform
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Mediapipe face detector
mp_face = mp.solutions.face_detection
face_detector = mp_face.FaceDetection(model_selection=1, min_detection_confidence=0.5)

def detect_and_classify_faces(frame):
    h, w = frame.shape[:2]
    results = face_detector.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    if results.detections:
        for detection in results.detections:
            bbox = detection.location_data.relative_bounding_box
            x1 = int(bbox.xmin * w)
            y1 = int(bbox.ymin * h)
            x2 = int((bbox.xmin + bbox.width) * w)
            y2 = int((bbox.ymin + bbox.height) * h)

            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            face = frame[y1:y2, x1:x2]
            if face.size == 0:
                continue

            face_input = transform(face).unsqueeze(0)
            with torch.no_grad():
                output = model(face_input)
                _, pred = torch.max(output, 1)
                
            label = classes[pred.item()]
            color = colors[pred.item()]

            # Draw box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Label background and text
            font_scale, thickness = 0.6, 2
            label_y = y1 - 10 if y1 > 20 else y2 + 20
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
            cv2.rectangle(frame, (x1, label_y - label_size[1] - 10),
                          (x1 + label_size[0] + 10, label_y), color, -1)
            cv2.putText(frame, label, (x1 + 5, label_y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)

    return frame

def run_on_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print("Image not found!")
        return

    output = detect_and_classify_faces(img)
    cv2.imshow("Mask Detection - Image", output)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def run_on_video(source=0):
    cap = cv2.VideoCapture(source)
    print("Press 'q' to quit")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        output_frame = detect_and_classify_faces(frame)
        cv2.imshow("Mask Detection", output_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def run_on_camera():
    cap = cv2.VideoCapture(0)
    print("Press 'q' to quit webcam.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        output = detect_and_classify_faces(frame)
        cv2.imshow("Mask Detection - Webcam", output)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# run_on_video("mask_video.mp4")

