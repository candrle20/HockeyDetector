# Hockey Player Object Detector


<img width="1267" alt="Screenshot 2023-12-20 at 5 43 51 PM" src="https://github.com/candrle20/HockeyDetector/assets/136523247/d340cd2d-efa7-4fc3-bab0-af5f53aa7710">




This repository contains two separate implementations of a hockey player detection system which was created by fine tuning the following pretrained models:

1. YOLOv8n
2. TensorFlow EfficientDet D1 640x640

Both models have been fine-tuned to detect objects of three classes: skaters, refs, goalies. These models were chosen because they were trained on the COCO dataset. The COCO dataset has 80 classes, one of which is people. Using models trained on this dataset enabled efficient transfer learning to recognize skaters, refs, and goalies (which are all similar to people) without the need for a large amount of training data.


## Model Info

YOLOv8

https://github.com/ultralytics/ultralytics

YOLOv8 Hockey Performance


<img width="1145" alt="Screenshot 2023-12-20 at 5 09 26 PM" src="https://github.com/candrle20/HockeyDetector/assets/136523247/8654e684-6fc2-4036-94e6-d09bcc51e20d">



Tensorflow 2 Model Zoo

https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md


## Dataset

The dataset used for to fine-tuning the models consisted of 120 images from various hockey broadcasts. I labeled these images using Roboflow's annotation tools. 

The data set can be seen here: https://universe.roboflow.com/hockeyplayer-model/hockey-player

## Next Steps

- Annotate a new testing data set to measure performance of these models vs off the shelf pretrained versions
- Implement boundry segmentation around ice surface to filter out false positives from detecting fans
- Train the model to differentiate between home vs away players

