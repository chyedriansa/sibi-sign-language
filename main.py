import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model
import json
import time
from typing import List, Tuple

class HandGestureRecognizer:
    def __init__(self, model_path: str, word_model_path: str, char_to_idx_path: str, idx_to_word_path: str):
        """Initialize the hand gesture recognizer with models and configurations."""
        # Load models
        self.gesture_model = load_model(model_path)
        self.word_model = load_model(word_model_path)
        
        # Load word prediction mappings
        with open(char_to_idx_path) as f:
            self.char_to_idx = json.load(f)
        with open(idx_to_word_path) as f:
            self.idx_to_word = {int(k): v for k, v in json.load(f).items()}
        
        # Configuration constants
        self.PADDING = 30
        self.MAX_SEQUENCE_LENGTH = 10
        self.PAD_TOKEN = '_'
        self.CONFIDENCE_THRESHOLD = 0.15
        self.INPUT_SIZE = (224, 224)
        self.labels = [chr(i) for i in range(65, 91)]  # A-Z
        
        # Initialize MediaPipe with optimized settings
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,  # Reduced from 2 for better performance
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5  # Added tracking confidence
        )
        
        # Performance tracking
        self.frame_count = 0
        self.fps_start_time = time.time()
        self.current_fps = 0
        
    def calculate_fps(self) -> float:
        """Calculate and return current FPS."""
        self.frame_count += 1
        if self.frame_count % 30 == 0:  # Update FPS every 30 frames
            current_time = time.time()
            self.current_fps = 30 / (current_time - self.fps_start_time)
            self.fps_start_time = current_time
        return self.current_fps
    
    def get_hand_bbox(self, hand_landmarks, width: int, height: int) -> Tuple[int, int, int, int]:
        """Extract bounding box coordinates from hand landmarks."""
        x_coords = [lm.x * width for lm in hand_landmarks.landmark]
        y_coords = [lm.y * height for lm in hand_landmarks.landmark]
        
        xmin, xmax = int(min(x_coords)), int(max(x_coords))
        ymin, ymax = int(min(y_coords)), int(max(y_coords))
        
        # Add padding with bounds checking
        xmin_p = max(xmin - self.PADDING, 0)
        ymin_p = max(ymin - self.PADDING, 0)
        xmax_p = min(xmax + self.PADDING, width)
        ymax_p = min(ymax + self.PADDING, height)
        
        return xmin_p, ymin_p, xmax_p, ymax_p
    
    def preprocess_hand_image(self, hand_crop: np.ndarray) -> np.ndarray:
        """Preprocess hand image for model prediction."""
        if hand_crop.size == 0:
            return None
        
        # Resize and normalize in one step
        resized = cv2.resize(hand_crop, self.INPUT_SIZE, interpolation=cv2.INTER_LINEAR)
        normalized = resized.astype(np.float32) / 255.0
        return np.expand_dims(normalized, axis=0)
    
    def predict_gesture(self, preprocessed_image: np.ndarray) -> Tuple[str, float]:
        """Predict gesture from preprocessed image."""
        if preprocessed_image is None:
            return None, 0.0
        
        prediction = self.gesture_model.predict(preprocessed_image, verbose=0)
        class_id = np.argmax(prediction)
        confidence = prediction[0][class_id]
        label = self.labels[class_id]
        
        return label, confidence
    
    def predict_top_k_words(self, prefix: str, k: int = 3) -> List[Tuple[str, float]]:
        """Predict top-k words based on character prefix."""
        if not prefix:
            return []
        
        prefix = prefix.lower()
        prefix_chars = list(prefix)
        
        # Pad sequence
        if len(prefix_chars) > self.MAX_SEQUENCE_LENGTH:
            prefix_chars = prefix_chars[:self.MAX_SEQUENCE_LENGTH]
        else:
            prefix_chars += [self.PAD_TOKEN] * (self.MAX_SEQUENCE_LENGTH - len(prefix_chars))
        
        # Convert to indices
        input_indices = [self.char_to_idx.get(char, self.char_to_idx[self.PAD_TOKEN]) 
                        for char in prefix_chars]
        input_array = np.array([input_indices])
        
        # Predict
        prediction_probs = self.word_model.predict(input_array, verbose=0)[0]
        top_k_indices = prediction_probs.argsort()[-k:][::-1]
        
        return [(self.idx_to_word[idx], prediction_probs[idx]) for idx in top_k_indices]
    
    def draw_predictions(self, frame: np.ndarray, bbox: Tuple[int, int, int, int], 
                        label: str, confidence: float, predicted_words: List[Tuple[str, float]]) -> None:
        """Draw bounding box and predictions on frame."""
        xmin_p, ymin_p, xmax_p, ymax_p = bbox
        
        # Draw bounding box
        cv2.rectangle(frame, (xmin_p, ymin_p), (xmax_p, ymax_p), (0, 255, 0), 2)
        
        # Draw gesture label
        cv2.putText(frame, f'{label} ({confidence*100:.1f}%)', 
                   (xmin_p, ymin_p - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        # Draw top predicted words
        if predicted_words:
            words_text = ', '.join([word for word, _ in predicted_words[:3]])
            cv2.putText(frame, f'Words: {words_text}', 
                       (xmin_p, ymin_p - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Process a single frame for hand detection and recognition."""
        height, width = frame.shape[:2]
        
        # Convert color space for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Get bounding box
                bbox = self.get_hand_bbox(hand_landmarks, width, height)
                xmin_p, ymin_p, xmax_p, ymax_p = bbox
                
                # Crop and preprocess hand region
                cropped_hand = frame[ymin_p:ymax_p, xmin_p:xmax_p]
                preprocessed = self.preprocess_hand_image(cropped_hand)
                
                if preprocessed is not None:
                    # Predict gesture
                    label, confidence = self.predict_gesture(preprocessed)
                    
                    if confidence > self.CONFIDENCE_THRESHOLD:
                        # Predict words
                        predicted_words = self.predict_top_k_words(label, k=3)
                        
                        # Draw results
                        self.draw_predictions(frame, bbox, label, confidence, predicted_words)
        
        return frame
    
    def run_camera(self, camera_id: int = 0) -> None:
        """Run the hand gesture recognition on camera feed."""
        cap = cv2.VideoCapture(camera_id)
        
        # Optimize camera settings
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer to minimize latency
        
        if not cap.isOpened():
            print("Error: Could not open camera")
            return
        
        print("Starting hand gesture recognition. Press 'ESC' to exit, 'q' to quit.")
        
        try:
            while True:
                success, frame = cap.read()
                if not success:
                    print("Failed to read from camera")
                    break
                
                # Process frame
                processed_frame = self.process_frame(frame)
                
                # Calculate and display FPS
                fps = self.calculate_fps()
                cv2.putText(processed_frame, f'FPS: {fps:.1f}', 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # Display frame
                cv2.imshow("Hand Gesture Recognition", processed_frame)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == 27 or key == ord('q'):  # ESC or 'q' to quit
                    break
                    
        except KeyboardInterrupt:
            print("\nInterrupted by user")
        finally:
            cap.release()
            cv2.destroyAllWindows()
    
    def __del__(self):
        """Cleanup resources."""
        if hasattr(self, 'hands'):
            self.hands.close()

def main():
    """Main function to run the hand gesture recognizer."""
    try:
        recognizer = HandGestureRecognizer(
            model_path='models/sibi-model-23052025-plat-arch-aug-v2.h5',
            word_model_path="word-predict/model_rnn_prediksi_kata.h5",
            char_to_idx_path="word-predict/char_to_idx.json",
            idx_to_word_path="word-predict/idx_to_word.json"
        )
        recognizer.run_camera()
    except FileNotFoundError as e:
        print(f"Error: Model file not found - {e}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()