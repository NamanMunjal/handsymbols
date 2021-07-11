import pickle
import cv2
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D,MaxPooling2D,Dense,Flatten
from tensorflow.keras.optimizers import Adam
from sklearn.utils import shuffle

files=["sci.pkl","pal.pkl","roc.pkl"]
features=[]
labels=[]

for i in range(0,len(files)):
	data=pickle.load(open(files[i],"rb"))
	for img in range(0,3001):
		features.append(data[img]/255)
		labels.append(i)


features,labels=shuffle(features,labels)
features,labels=np.array(features),np.array(labels)
features=features.reshape(len(features),112,63,1)

model=Sequential()

model.add(Conv2D(1024,activation='relu',kernel_size=(4,4),input_shape=(112,63,1)))
model.add(MaxPooling2D(pool_size=(4,4)))
model.add(Conv2D(256,activation='relu',kernel_size=(4,4)))
model.add(MaxPooling2D(pool_size=(4,4)))
model.add(Flatten())
model.add(Dense(64,activation='relu'))
model.add(Dense(3,activation='softmax'))

model.compile(loss="sparse_categorical_crossentropy",optimizer=Adam(learning_rate=1e-3),metrics=['accuracy'])
model.summary()
model.fit(features,labels,epochs=2,batch_size=16)
model.save("model.h5")
