# handsymbols
Recognizing hand symbols like rock,paper and scissor using CNN

There were mainly 2 steps involved:
1) Preparing data
2) Creating the AI using a CNN

To make the data feasible for the AI to read I subtracted the background from the image making only the hand visible

![rock](https://user-images.githubusercontent.com/65707802/125201218-bdf1bd00-e28b-11eb-862b-46524de41f1b.jpg)

After putting all this data in pickle files I made a CNN model to predict the hand symbol and got an accuracy of 96% all thanks to the background subtraction

![image](https://user-images.githubusercontent.com/65707802/125201286-0d37ed80-e28c-11eb-9bdb-50ced730db45.png)
