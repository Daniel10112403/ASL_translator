# ASL Translator


*Transforming communication through sign language recognition.*

## Overview

The **ASL Translator** is an innovative project designed to bridge communication gaps by translating American Sign Language (ASL) alphabet into text . Using machine learning and computer vision, this tool aims to empower individuals who are deaf or hard of hearing, as well as those who communicate with them, by providing real-time translation of sign language gestures.

This project currently focuses on recognizing a subset of ASL gestures using image-based input. My vision is to expand its capabilities to support more sign languages, improve accuracy, and integrate video-based recognition for real-time applications.

---

## Features

- **ASL Gesture Recognition**: Translates a predefined set of ASL hand gestures into text.
- **Image-Based Input**: Processes static images of hand signs using computer vision techniques.
- **Machine Learning Model**: Utilizes a trained ML model to classify ASL gestures with high accuracy.
- **Open Source**: Licensed under the MIT License, encouraging collaboration and contributions.

---


## Next Steps
We’re excited about the future of the ASL Translator and have several goals to enhance its functionality and impact:
1. **Adapt to Ecuadorian Sign Language (LSE):**
    -Research and collect a dataset of Ecuadorian Sign Language gestures.
    -Modify the existing model to recognize LSE gestures, potentially by fine-tuning on a new dataset.
    -Collaborate with local communities in Ecuador to ensure cultural accuracy and relevance.
2. **Explore Video-Trained ML Algorithms:**
    -Transition from static image recognition to video-based recognition for real-time translation.
    -Investigate state-of-the-art video-trained models like 3D CNNs or transformer-based architectures (e.g.    TimeSformer) to capture temporal dynamics of sign language gestures.
    -Implement a pipeline for processing live video streams from a webcam.
3. **Improve Accuracy and Expand Vocabulary:**
    -Increase the number of recognized gestures to cover the full ASL alphabet and common words.
    -Enhance model accuracy by incorporating more diverse training data.
4. **Develop a Mobile App:**
    -Create a mobile application to make the translator accessible on smartphones.
    -Integrate with device cameras for on-the-go sign language translation.
