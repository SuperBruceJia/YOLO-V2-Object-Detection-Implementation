# YOLO-V2-Object-Detection-Implementation

Author: Shuyue Jia

Date: December of 2017

## The codes are used to implement Keras h5 model to a certain Image or video. 

The main program is "detection_main.py" based on <= TensorFlow 1.12 with CUDA 9.0 (or TensorFlow 1.13 with CUDA 10.0) and Keras, scipy, OpenCV. Specifically, the codes need to be run at a Python 3.5 environment!!!

## By doing this, u need to create an Python 3.5 environment via Anaconda Conda:

```
conda create --name yolo python=3.5 numpy
```

Then, in the terminal of Anaconda Prompt:

```
activate yolo
```

## Under the yolo environment, install the needed Python packages

```
(If u have CUDA 10.0): pip install --upgrade --force-reinstall tensorflow-gpu==1.13.1 --user

(If u have CUDA 9.0): pip install --upgrade --force-reinstall tensorflow-gpu==1.12.0 --user

(If u don't have a GPU): pip install --upgrade --force-reinstall tensorflow==1.12.0 --user

pip install pandas

pip install scipy==1.0.0
```

## Please download official trained h5 model from Google Driver: 

Please download the trained model from [here](https://drive.google.com/file/d/11pFtogeYDPC6iMi7w0ZQwaeYuui-TRgV/view?usp=sharing).

And place it in the "model_data" folder.

## Please put your images in the "images" folder and after the detection program, the output Image will be saved in the "out" folder. 

## Objects could be detected:

```
person
bicycle
car
motorbike
aeroplane
bus
train
truck
boat
traffic light
fire hydrant
stop sign
parking meter
bench
bird
cat
dog
horse
sheep
cow
elephant
bear
zebra
giraffe
backpack
umbrella
handbag
tie
suitcase
frisbee
skis
snowboard
sports ball
kite
baseball bat
baseball glove
skateboard
surfboard
tennis racket
bottle
wine glass
cup
fork
knife
spoon
bowl
banana
apple
sandwich
orange
broccoli
carrot
hot dog
pizza
donut
cake
chair
sofa
pottedplant
bed
diningtable
toilet
tvmonitor
laptop
mouse
remote
keyboard
cell phone
microwave
oven
toaster
sink
refrigerator
book
clock
vase
scissors
teddy bear
hair drier
toothbrush
```