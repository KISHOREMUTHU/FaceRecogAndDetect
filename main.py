import cv2
import numpy as np
import os


def distance(v1, v2):
    return np.sqrt(((v1-v2)**2).sum())

# KNN Code


def knn(train, test, k=5):
    dist =[]
    for i in range(train.shape[0]):
         ix =train[i, :-1]
         iy =train[i, -1]
         d = distance(test, ix)
         dist.append([d, iy])
    dk =sorted(dist, key=lambda x: x[0])[:k]
    labels = np.array(dk)[:, -1]
    output = np.unique(labels, return_counts=True)
    index = np.argmax(output[1])
    return output[0][index]

#Data Preparation
cap=cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
class_id =0
names ={}
dataset_path = 'C:\\Users\\dell\\PycharmProjects\\'
face_data =[]
labels = []
for fx in os.listdir(dataset_path):
     if fx.endswith('.npy'):
            names[class_id]=fx[:-4]

            data_item =np.load(dataset_path+fx)
            face_data.append(data_item)
            target = class_id * np.ones((data_item.shape[0],))

            class_id +=1
            labels.append(target)

face_dataset = np.concatenate(face_data, axis=0)
face_labels = np.concatenate(labels, axis=0).reshape((-1,1))
train_set =np.concatenate((face_dataset, face_labels), axis=1)
print(train_set.shape)

while True:
      ret, frame = cap.read()
      if ret==False :
          continue
      #cv2.imshow("frame1",frame)
      gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
      faces = face_cascade.detectMultiScale(gray_frame, 1.3, 5)
      if (len(faces) == 0):
          continue
      faces =sorted(faces, key=lambda f: f[2] * f[3])
      # We have to pick the face having the largest area
      for face in faces[-1:]:
          x, y, w, h = face


          #cv2.imshow("frame2", frame)
          offset = 10
          face_section = gray_frame[y - offset:y + h + offset, x - offset:x + w + offset]
          face_section = cv2.resize(face_section, (100, 100))

          #predict the output using knn
          out =knn(train_set,face_section.flatten())
          # Display the output on the screen
          pred_name = names [int(out)]
          cv2.putText(gray_frame,pred_name,(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2,cv2.LINE_AA)


          cv2.rectangle(frame, (x, y), (x+w,y+h), (0, 255, 255), 2)

          # cv2.imshow("frame2",frame)

          cv2.imshow("gray", gray_frame)
      key_press = cv2.waitKey(1) & 0xFF
      if key_press == ord('q'):
          break

cap.release()
cv2.waitKey(0)
cv2.destroyAllWindows()