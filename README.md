# Vehicle Detection Project

[//]: # (Image References)

[image0]: ./img/ssd.png "Pipeline"

<p align="center">
 <a href="https://youtu.be/JlFFbvWWsHg"><img src="./output_video/project_video.gif" alt="Overview" width="50%" height="50%"></a>
 <br>Qualitative results. (click for full video)
</p>


### Abstract

The goal of the project was to develop a pipeline to reliably detect cars given a video from a roof-mounted camera

 [SSD deep network](https://arxiv.org/pdf/1512.02325.pdf) for detection, thresholds on detection confidence and label to discard false positive 
 
*That said, let's go into details!*

### Computer Vision on Steroids, a.k.a. Deep Learning

#### 1. SSD (*Single Shot Multi-Box Detector*) network

 - the network performs detection and classification in a single pass, and natively goes in GPU (*is fast*)
 - there is no more need to tune and validate hundreds of parameters related to the phase of feature extraction (*is robust*)
 - being the "car" class in very common, various pretrained models are available in different frameworks (Keras, Tensorflow etc.) that are already able to nicely distinguish this class of objects (*no need to retrain*)
 - the network outputs a confidence level along with the coordinates of the bounding box, so we can decide the tradeoff precision and recall just by tuning the confidence level we want (*less false positive*) 
 
The whole pipeline has been adapted to the make use of SSD network in file [`main_ssd.py`](main_ssd.py).

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](https://youtu.be/JlFFbvWWsHg)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

The overal structure of SSD is following:
![alt text][image0]


---


### Acknowledgments

Implementation of [Single Shot MultiBox Detector](https://arxiv.org/pdf/1512.02325.pdf) was borrowed from [this repo](https://github.com/rykov8/ssd_keras) .
