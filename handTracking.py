import cv2
import mediapipe as mp
from mediapipe.tasks.python import vision
import torch
import torch.nn as nn
from torchvision import models, transforms
from collections import deque, Counter
import torch.nn.functional as F


MODEL_PATH = "hand_landmarker.task"

pred_buffer = deque(maxlen=5)

# device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# recreate model
model = models.resnet18(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, 29)

# load weights
model.load_state_dict(torch.load("asl_model_3(new_dataset).pth", map_location=device))
model.to(device)
model.eval()

# transforms (same as training)
transform = transforms.Compose(
    [
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225]),
    ]
)

# class labels (IMPORTANT: match Kaggle order)
classes = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N',
           'O','P','Q','R','S','T','U','V','W','X','Y','Z','del','nothing','space']

options = vision.HandLandmarkerOptions(
    base_options=mp.tasks.BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=vision.RunningMode.VIDEO,
    num_hands=2
)

cap = cv2.VideoCapture(0)
timestamp = 0

with vision.HandLandmarker.create_from_options(options) as landmarker:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        result = landmarker.detect_for_video(mp_image, timestamp)
        timestamp += 1

        if result.hand_landmarks:
            h, w, _ = frame.shape

            for hand_landmarks in result.hand_landmarks:
                xs = [int(lm.x * w) for lm in hand_landmarks]
                ys = [int(lm.y * h) for lm in hand_landmarks]

                cx = int(sum(xs) / len(xs))
                cy = int(sum(ys) / len(ys))

                size = max(max(xs) - min(xs), max(ys) - min(ys)) + 40

                x_min = max(0, cx - size // 2)
                y_min = max(0, cy - size // 2)
                x_max = min(w, cx + size // 2)
                y_max = min(h, cy + size // 2)

                hand_crop = frame[y_min:y_max, x_min:x_max]

                if hand_crop.size != 0:
                    hand_crop = cv2.cvtColor(hand_crop, cv2.COLOR_BGR2RGB)
                    input_tensor = transform(hand_crop).unsqueeze(0).to(device)
                    with torch.no_grad():
                        output = model(input_tensor)

                        probs = F.softmax(output, dim=1)
                        conf, pred = torch.max(probs, 1)

                        pred = pred.item()

                    # confidence filter FIRST
                    if conf.item() < 0.6:
                        label = "..."
                    else:
                        pred_buffer.append(pred)

                        stable_pred = Counter(pred_buffer).most_common(1)[0][0]
                        label = classes[stable_pred]

                    # label = classes[pred]

                    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

                    cv2.putText(frame, label, (x_min, y_min - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # ALWAYS show frame
        cv2.imshow("Hands", frame)

        # ESC to quit
        if cv2.waitKey(1) & 0xFF == 27:
            break
