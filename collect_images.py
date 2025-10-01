import os
import cv2

# Letters (classes) jinke liye data collect karna hai
LETTERS = ['A', 'B', 'C', 'F', 'I', 'K', 'L', 'O', 'U', 'V', 'W', 'Y']

DATA_DIR = './data'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

dataset_size = 100  # Har letter ke liye images ki count

cap = cv2.VideoCapture(0)

for letter in LETTERS:
    # Folder create for each letter
    letter_dir = os.path.join(DATA_DIR, letter)
    if not os.path.exists(letter_dir):
        os.makedirs(letter_dir)

    print(f'Collecting data for letter: {letter}')

    # Wait until user presses Q to start capturing
    while True:
        ret, frame = cap.read()
        cv2.putText(frame, f'Letter: {letter} | Press "Q" to start',
                    (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
                    1.2, (0, 255, 0), 3, cv2.LINE_AA)
        cv2.imshow('frame', frame)
        if cv2.waitKey(25) == ord('q'):
            break

    counter = 0
    while counter < dataset_size:
        ret, frame = cap.read()
        cv2.imshow('frame', frame)
        cv2.waitKey(25)
        cv2.imwrite(os.path.join(letter_dir, f'{counter}.jpg'), frame)
        counter += 1

cap.release()
cv2.destroyAllWindows()
