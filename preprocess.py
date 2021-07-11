import pickle
import cv2
import numpy

files=["pal.pkl","roc.pkl","sci.pkl"]

for file in files:
	data=pickle.load(open(file,"rb"))
	imgs=[]
	for img in data:
		imgs.append(cv2.resize(img,(112,63)))
	pickle.dump(imgs,open(file,"wb"))
