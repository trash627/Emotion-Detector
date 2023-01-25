import cv2
import sys
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

frame_per_second = 30
emotionlLabels = {0: "Angry", 1: "Disgusted", 2: "Fearful",
                  3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}


cascPath = "haarcascade_frontalface_default.xml"
# model = "Live-Emotion-Detection/inputs/models/modelV1.h5"
model = "model_resnet1.h5"
model = load_model(model)


faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + cascPath)


def drawFrame(frame):

    if frame is not None:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = faceCascade.detectMultiScale(gray)

        # Draw a rectangle around the faces
        if len(faces) != 0:
            for (x, y, w, h) in [faces[0]]:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 250, 250), 4)

                H, W, _ = frame.shape
                x2 = x + int(1.0 * w)
                y2 = y + int(1.0 * h)

                faceImg = gray[y:y2, x:x2].copy()
                # changing size to feed cropped face to model
                faceImg = cv2.resize(faceImg, (48, 48))
                faceImg = faceImg.reshape([-1, 48, 48, 1])
                faceImg = np.multiply(faceImg, 1.0 / 255.0)

                classifiedNumb = model.predict(faceImg)
                predicted_label = int(np.argmax(classifiedNumb[0]))
                cv2.putText(frame, str(emotionlLabels[predicted_label]), (
                    x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 25*w/W, (0, 255, 0), 3)


def videoMaker(inputFile=None):

    w, h = None, None
    if inputFile == None:

        imageFrame = cv2.VideoCapture(0)
        while True:
            # Capture frame-by-frame
            ret, frame = imageFrame.read()

            drawFrame(frame)

            # Display the resulting frame

            cv2.imshow('Video', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            # When everything is done, release the capture

            # Checking to close window if 'c' is pressed
            key = cv2.waitKey(25)
            if key == ord('c'):
                break
        imageFrame.release()
        cv2.destroyAllWindows()

    else:
        imageFrame = cv2.VideoCapture(inputFile)
        while True:
            ret, frame = imageFrame.read()
            if w is None:
                # Setting up parameters for output mp4
                h, w, _ = frame.shape
                fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
                writer = cv2.VideoWriter(
                    'output.mp4', fourcc, frame_per_second, (w, h))
            if frame is None:
                break
            # grawing detected face and emotion over the frame
            drawFrame(frame)
            writer.write(frame)  # saving frame into output video
        writer.release()


def main():
    # videoMaker('FaceTest.mp4')

    videoMaker()


if __name__ == '__main__':
    main()
