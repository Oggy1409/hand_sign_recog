✋ Hand Sign Recognition

A real-time hand sign recognition system built with Python, OpenCV, Mediapipe, and Random Forest Classifier.
The system captures hand landmarks, trains a machine learning model, and predicts gestures live using the webcam feed.

🚀 Features

✅ Real-time hand detection using Mediapipe

✅ Gesture classification with Random Forest Classifier

✅ High accuracy (~97% on custom dataset)

✅ OpenCV-based UI (bounding boxes, prediction overlay)

✅ Custom dataset support – create your own hand sign set

🛠️ Tech Stack

Python 3.x

OpenCV – real-time video & visualization

Mediapipe – extracting hand landmarks

Scikit-learn – machine learning classifier

Pickle – saving & loading trained model

NumPy – dataset handling

⚙️ How It Works
1️⃣ Dataset Creation

Run: python create_dataset.py

Capture hand landmarks for each gesture

Dataset is saved as dataset.pickle

2️⃣ Train the Classifier

Run: python train_classifier.py

Trains Random Forest Classifier

Model saved as model.p

3️⃣ Real-time Inference

Run: python inference_classifier.py

Opens webcam and predicts gestures in real-time

Displays prediction on video feed with bounding boxes

📊 Accuracy

Training Accuracy: 97.75%

Best performance with good lighting and clear hand gestures

🔮 Future Improvements

🤖 Replace Random Forest with Deep Learning (CNN, LSTM)

✋ Support multi-hand recognition

😀 Add emoji/text-to-speech output

🖥️ Build a user-friendly GUI (Tkinter / PyQT)


📌 Example Output

<img width="805" height="620" alt="Screenshot 2025-10-01 131235" src="https://github.com/user-attachments/assets/cbbee9ca-2f95-41a9-8c02-510fcd8c72ca" />




