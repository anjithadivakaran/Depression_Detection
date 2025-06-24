# Depression_Detection
A deep learning model that combines text and image data from social media timelines to detect signs of depression in users.

## Overview
This project implements a multimodal timeline-based classifier for depression detection. Each user's timeline includes up to 50 posts with both textual and visual content. The model uses BERT for text encoding and ResNet for image encoding, followed by GRU-based temporal fusion and a final classifier. The Goal is to 

## Dataset Used
Available: https://pan.baidu.com/s/1j5M1PWNYVgM9HmmkY60uyQ?pwd=miww

## Model Architecture


## Results

| Metric        | Test   |
|---------------|--------|
| Accuracy      | 97.37% |
| F1 Score      | 97.14% |

These are preliminary results on a balanced 500-user dataset.

