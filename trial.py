import cv2
import pickle
import numpy as np
from tensorflow.keras.models import load_model
from time import sleep
video=cv2.VideoCapture(0)
kernel=np.ones((8,8),np.uint8)
ret,frame=video.read()
i=0
model=load_model("model.h5")
images=[]
dic={0:"sci",1:"pal",2:"roc"}
run=True
while run:
	if i==0:
		ret,bg=video.read()
		bg=cv2.cvtColor(bg,cv2.COLOR_BGR2GRAY)
		i+=1
		print("5 seconds")
		sleep(5)
	else:
		ret,img2=video.read()
		img2=cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
		mask=cv2.absdiff(img2,bg)
		thresh=cv2.threshold(mask,25,255,cv2.THRESH_BINARY)[1]
		thresh=cv2.erode(thresh,kernel)
		cv2.imshow("subtracted",thresh)
		thresh=cv2.resize(thresh,(112,63))
		thresh=thresh.reshape(1,112,63,1)
		if cv2.countNonZero(thresh) ==0:
			print("None")
		else:
			print(dic[np.argmax(model.predict(thresh))])
		cv2.waitKey(1)
		
	if cv2.waitKey(1) & 0xFF == ord('q'):
		run=False