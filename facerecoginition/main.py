import cv2
import numpy as np

cap =cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
face_data =[]
#dataset_path="C:\Users\dell\PycharmProjects"
file_name = input("Enter the name of the person : ")

while True :
      ret , frame = cap.read()
      if ret == False :
          continue
      #cv2.imshow("frame1",frame)
      gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
      faces = face_cascade.detectMultiScale(gray_frame, 1.3, 5)
      if (len(faces) == 0):
          continue
      faces = sorted(faces, key=lambda f: f[2] * f[3])
      # We have to pick the face having the largest area
      for face in faces[-1:]:
          x, y, w, h = face
          cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)
          cv2.imshow("frame2", frame)
          offset = 10
          face_section = gray_frame[y - offset:y + h + offset, x - offset:x + w + offset]
          face_section = cv2.resize(face_section, (100, 100))
          face_data.append(face_section)
          print(len(face_section))
          # cv2.imshow("frame2",frame)

          cv2.imshow("gray", gray_frame)
      key_press = cv2.waitKey(1) & 0xFF
      if key_press == ord('q'):
          break
face_data = np.asarray(face_data)
face_data = face_data.reshape((face_data.shape[0], -1))
print(face_data.shape)
np.save('C:\\Users\\dell\\PycharmProjects\\' + file_name + '.npy', face_data)
print("Data saved")

cap.release()
cv2.waitKey(0)
cv2.destroyAllWindows()