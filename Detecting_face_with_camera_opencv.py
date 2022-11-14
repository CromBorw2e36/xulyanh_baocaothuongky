import cv2
import numpy as np

# hàm phát hiện đối tượng để học máy
# khuôn mặt
face_cascade = cv2.CascadeClassifier('./lib/haarcascade_frontalface_default.xml')
# đôi mắt
eye_cascade = cv2.CascadeClassifier('./lib/haarcascade_eye.xml')
# cái miệng
# smile_cascade = cv2.CascadeClassifier('./lib/haarcascade_smile.xml')
smile_cascade = cv2.CascadeClassifier('./lib/haarcascade_mcs_mouth.xml')
#   tai trái  - phải
left_ear_cascade = cv2.CascadeClassifier('./lib/haarcascade_mcs_leftear.xml')
right_ear_cascade = cv2.CascadeClassifier('./lib/haarcascade_mcs_rightear.xml')
# cái mũi
nose_cascade = cv2.CascadeClassifier('./lib/haarcascade_mcs_nose.xml')

if nose_cascade.empty():
  raise IOError('Unable to load the nose cascade classifier xml file')

cap = cv2.VideoCapture(0)

# set size windows
ds_factor = 1.5

while True:
    ret, frame = cap.read()
    frame = cv2.resize(frame, None, fx=ds_factor, fy=ds_factor, interpolation=cv2.INTER_AREA)

    face = face_cascade.detectMultiScale(frame, scaleFactor=1.2, minNeighbors=4)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    for (x, y, w, h) in face:
        # xửa lý vùng khuôn mặt
        img_rect = cv2.rectangle(frame, (x, y), (x + w, y + h), (247, 164, 164), 5)

        text = "Face"
        org = (x + w, y - 10)
        fontFace = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 0.9
        color = (0, 0, 0)
        lineType = cv2.LINE_4

        img_text = cv2.putText(img_rect, text, org, fontFace, fontScale, color, lineType)

        roi_gray = gray[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]

        # Phát hiện các đối tượng (được qui định trong Classifier) với các kích thước khác nhau có trong bức ảnh
        eyes = eye_cascade.detectMultiScale(roi_gray, 1.3, 5)
        smile = smile_cascade.detectMultiScale(roi_gray, 5, 5)
        nose = nose_cascade.detectMultiScale(roi_gray, 1.3, 5)
        left_ear = left_ear_cascade.detectMultiScale(frame, 5, 5)
        right_ear = right_ear_cascade.detectMultiScale(frame, 5, 5)

        # tai trái
        for (lex, ley, lew, leh) in left_ear:
            cv2.rectangle(gray, (lex, ley), (lex + lew, ley + leh), (0, 0, 255), 3)
            break
        # tai phải
        for (rex, rey, rew, reh) in right_ear:
            cv2.rectangle(gray, (rex, rey), (rex + rew, rey + reh), (0, 0, 255), 3)
            break
        # mắt
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (254, 190, 140), 5)
        # miệng
        for (sx, sy, sw, sh) in smile:
            cv2.rectangle(roi_color, (sx, sy), (sx + sw, sy + sh), (255, 88, 88), 5)
            break
        # mũi
        for (sx, sy, sw, sh) in nose:
            cv2.rectangle(roi_color, (sx, sy), (sx + sw, sy + sh), (182, 226, 161), 5)
            break

    cv2.imshow('Face Detector', frame)

    c = cv2.waitKey(1)
    if c == 27:
        break

cap.release()
cv2.destroyAllWindows()