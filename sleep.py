import cv2
import playsound


face_cascade = cv2.CascadeClassifier('C:/Users/ashlaeteng/anaconda3/pkgs/opencv-3.2.0-np112py36_0/share/OpenCV/haarcascades/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('C:/Users/ashlaeteng/anaconda3/pkgs/opencv-3.2.0-np112py36_0/share/OpenCV/haarcascades/haarcascade_eye.xml')
face_cascade.load('/Users/ashlaeteng/anaconda3/pkgs/opencv-3.2.0-np112py36_0/share/OpenCV/haarcascades/haarcascade_frontalface_default.xml')
eye_cascade.load('/Users/ashlaeteng/anaconda3/pkgs/opencv-3.2.0-np112py36_0/share/OpenCV/haarcascades/haarcascade_eye.xml')


cap = cv2.VideoCapture(0)
counter = 0
while(True):

    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)


    for (x,y,w,h) in faces:
        img = cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+int(h/2), x:x+w]
        roi_color = img[y:y+int(h/2), x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        if len(eyes) == 0:
            counter += 1
            if counter > 5:
                playsound.playsound('sleep_file.mp3', True)
        else:
            counter = 0
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
    cv2.imshow('frame',frame)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
