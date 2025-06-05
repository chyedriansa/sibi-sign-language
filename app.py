from flask import Flask, render_template, Response
import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
import base64
import json

# Initialize Flask app
app = Flask(__name__)

# Load models
hand_model = load_model('models/sibi-model-23052025-plat-arch-aug-v2.h5')
word_model = load_model("word-predict/model_rnn_prediksi_kata.h5")

# Load mappings
with open("word-predict/char_to_idx.json") as f:
    char_to_idx = json.load(f)
with open("word-predict/idx_to_word.json") as f:
    idx_to_word = {int(k): v for k, v in json.load(f).items()}

# Constants
LABELS = [chr(i) for i in range(65, 91)]
MAX_SEQ_LEN = 10
PAD_TOKEN = '_'
PADDING = 30

# MediaPipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=1,
                       min_detection_confidence=0.7)

# Word prediction function
def predict_words(prefix, k=3):
    prefix = (list(prefix.lower()) + [PAD_TOKEN] * MAX_SEQ_LEN)[:MAX_SEQ_LEN]
    indices = [char_to_idx.get(c, char_to_idx[PAD_TOKEN]) for c in prefix]
    input_array = np.array([indices])
    probs = word_model.predict(input_array, verbose=0)[0]
    top_k = probs.argsort()[-k:][::-1]
    return [(idx_to_word[i], float(probs[i])) for i in top_k]

# Frame processing
def process_frame(frame):
    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if results.multi_hand_landmarks:
        for landmarks in results.multi_hand_landmarks:
            x = [lm.x * w for lm in landmarks.landmark]
            y = [lm.y * h for lm in landmarks.landmark]
            xmin, xmax = int(min(x)), int(max(x))
            ymin, ymax = int(min(y)), int(max(y))
            xmin_p, ymin_p = max(xmin - PADDING, 0), max(ymin - PADDING, 0)
            xmax_p, ymax_p = min(xmax + PADDING, w), min(ymax + PADDING, h)

            cropped = frame[ymin_p:ymax_p, xmin_p:xmax_p]
            resized = cv2.resize(cropped, (224, 224)).astype('float32') / 255.0
            input_img = np.expand_dims(resized, axis=0)

            preds = hand_model.predict(input_img)
            class_id = np.argmax(preds)
            label = LABELS[class_id]
            conf = float(preds[0][class_id])

            if conf > 0.1:
                predicted = predict_words(label)
                cv2.rectangle(frame, (xmin_p, ymin_p), (xmax_p, ymax_p), (0, 255, 0), 2)
                cv2.putText(frame, f'{label} ({conf*100:.1f}%)', (xmin_p, ymin_p - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                return frame, {
                    "detected": True,
                    "label": label,
                    "predicted_words": predicted,
                    "confidence": conf,
                    "x_min": xmin_p,
                    "y_min": ymin_p,
                    "x_max": xmax_p,
                    "y_max": ymax_p
                }
    return frame, {"detected": False}

# Stream video frames
def get_frame():
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        processed_frame, prediction = process_frame(frame)
        _, jpeg = cv2.imencode('.jpg', processed_frame)
        encoded_img = base64.b64encode(jpeg.tobytes()).decode('utf-8')
        yield f"data: {json.dumps({'image': encoded_img, 'prediction': prediction})}\n\n"
    cap.release()

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(get_frame(), mimetype='text/event-stream')

@app.route('/shutdown', methods=['POST'])
def shutdown():
    return "Capture release handled in generator."

# Run app
if __name__ == '__main__':
    app.run(debug=True, threaded=True)
