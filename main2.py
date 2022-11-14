import cv2
#đọc hình ảnh
ListImages = [
    "./Images/h5-bui-thi-my-duyen-1587808490674636393748.jpg",
    "./Images/175337396_2917917531830460_8008229113997594091_n.jpg",
    "./Images/vzV31yE.jpg",
    "Images/LoanDinh.jpg",
    "Images/LoanDinh2.jpg",
    "./Images/dan-ong-thuong-cam-thay-ho-co-the-la-chinh-minh-khi-hen-ho-voi-mot-co-gai-binh-thuong-444281.jpg",
    "./Images/Anh-gai-xinh-Viet-Nam.jpg"
]

img  = cv2.imread(ListImages[3], cv2.IMREAD_COLOR)

img_Resize = cv2.resize(img, (600, 600))

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

face = face_cascade.detectMultiScale(img_Resize, scaleFactor = 1.2, minNeighbors = 4)
gray = cv2.cvtColor(img_Resize, cv2.COLOR_BGR2GRAY)


for (x, y, w, h) in face:
    # xửa lý vùng khuôn mặt
    img_rect = cv2.rectangle(img_Resize, (x, y), (x + w, y + h), (247, 164, 164), 5,cv2.LINE_4)

    text = "Face"
    org = (x + w,y - 10)
    fontFace = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 0.9
    color = (0, 0, 0)
    lineType = cv2.LINE_4

    img_text = cv2.putText(img_rect, text, org, fontFace, fontScale, color, lineType)

    roi_gray = gray[y:y + h, x:x + w]
    roi_color = img_Resize[y:y + h, x:x + w]

    eyes = eye_cascade.detectMultiScale(roi_gray, 1.3, 5)
    smile = smile_cascade.detectMultiScale(roi_gray, 5, 5)
    nose = nose_cascade.detectMultiScale(roi_gray, 1.3, 5)

    left_ear = left_ear_cascade.detectMultiScale(roi_gray, 5, 5)
    right_ear = right_ear_cascade.detectMultiScale(roi_gray, 5, 5)

    # tai trái
    for (lex, ley, lew, leh) in left_ear:
        cv2.rectangle(roi_color, (lex, ley), (lex + lew, ley + leh), (0, 0, 255), 3)
        break
    # tai phải
    for (rex, rey, rew, reh) in right_ear:
        cv2.rectangle(roi_color, (rex, rey), (rex + rew, rey + reh), (0, 0, 255), 3)
        break
    # mắt
    for (ex, ey, ew, eh) in eyes:
        cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (254, 190, 140), 5)
    # miệng
    for (sx, sy, sw, sh) in smile:
        cv2.rectangle(roi_color, (sx, sy), (sx + sw, sy + sh), (255, 251, 193), 5)
        break
    # mũi
    for (sx, sy, sw, sh) in nose:
        cv2.rectangle(roi_color, (sx, sy), (sx + sw, sy + sh), (182, 226, 161), 5)
        break

cv2.imshow("Eyes Detected", img_Resize)

cv2.waitKey(0)

cv2.destroyAllWindows()