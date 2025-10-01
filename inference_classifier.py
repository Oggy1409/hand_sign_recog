import pickle
import cv2
import mediapipe as mp
import numpy as np
import warnings
import platform

warnings.filterwarnings('ignore')

class HandSignRecognizer:
    def __init__(self, model_path='./model.p'):
        # Load model
        self.model = None
        self.labels_dict = {
            0: 'A', 1: 'B', 2: 'C', 3: 'F', 4: 'I',
            5: 'K', 6: 'L', 7: 'O', 8: 'U', 9: 'V',
            10: 'W', 11: 'Y'
        }

        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )

        self.frame_count = 0
        self.prediction_count = 0
        self.error_count = 0

        self.load_model(model_path)

    def load_model(self, path):
        try:
            with open(path, 'rb') as f:
                self.model = pickle.load(f)['model']
            # Windows compatibility
            if hasattr(self.model, 'n_jobs') and platform.system() == "Windows":
                self.model.n_jobs = 1
            print(f"✓ Model loaded successfully")
        except Exception as e:
            print(f"❌ Failed to load model: {e}")

    def extract_landmarks(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)

        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]

            x_ = [lm.x for lm in hand_landmarks.landmark]
            y_ = [lm.y for lm in hand_landmarks.landmark]

            data_aux = []
            for lm in hand_landmarks.landmark:
                data_aux.append(lm.x - min(x_))
                data_aux.append(lm.y - min(y_))

            if len(data_aux) != 42:
                return None, None
            return np.array(data_aux), hand_landmarks
        return None, None

    def predict(self, landmarks):
        try:
            features = landmarks.reshape(1, -1)
            pred = self.model.predict(features)[0]
            self.prediction_count += 1
            return self.labels_dict.get(pred, '?')
        except Exception:
            self.error_count += 1
            return '?'

    def draw_ui(self, frame, predicted_letter, hand_landmarks):
        H, W, _ = frame.shape

        # Draw hand landmarks + bounding box
        if hand_landmarks:
            self.mp_drawing.draw_landmarks(
                frame, hand_landmarks,
                self.mp_hands.HAND_CONNECTIONS,
                self.mp_drawing_styles.get_default_hand_landmarks_style(),
                self.mp_drawing_styles.get_default_hand_connections_style()
            )

            x_list = [lm.x for lm in hand_landmarks.landmark]
            y_list = [lm.y for lm in hand_landmarks.landmark]
            x1, y1 = int(min(x_list) * W) - 20, int(min(y_list) * H) - 20
            x2, y2 = int(max(x_list) * W) + 20, int(max(y_list) * H) + 20
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 0), 3)

            # Prediction box above bounding box
            if predicted_letter and predicted_letter != "?":
                cv2.rectangle(frame, (x1, y1 - 60), (x1 + 120, y1), (0, 200, 0), -1)
                cv2.putText(frame, predicted_letter, (x1 + 15, y1 - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 4)

        # --- Side Info Panel ---
        panel_w = 260
        cv2.rectangle(frame, (W - panel_w - 10, 10), (W - 10, 140), (30, 30, 30), -1)

        cv2.putText(frame, "📊 Stats", (W - panel_w + 20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(frame, f"Frames: {self.frame_count}", (W - panel_w + 20, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(frame, f"Predictions: {self.prediction_count}", (W - panel_w + 20, 95),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
        cv2.putText(frame, f"Errors: {self.error_count}", (W - panel_w + 20, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 100, 255), 1)

        # --- Controls Footer ---
        cv2.rectangle(frame, (0, H - 40), (W, H), (50, 50, 50), -1)
        cv2.putText(frame, "Controls: Q = Quit | R = Reset Stats",
                    (20, H - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1)

        return frame

    def run(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ Cannot open camera")
            return

        print("✓ Camera opened, show your hand to detect letters!")

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            self.frame_count += 1
            frame = cv2.flip(frame, 1)

            landmarks, hand_landmarks = self.extract_landmarks(frame)
            predicted_letter = self.predict(landmarks) if landmarks is not None else None

            frame = self.draw_ui(frame, predicted_letter, hand_landmarks)

            cv2.imshow("Hand Sign Recognition", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                self.frame_count = self.prediction_count = self.error_count = 0
                print("Stats reset!")

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    recognizer = HandSignRecognizer()
    recognizer.run()
