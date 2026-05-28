# kc-lib

Machine learning for Python beginners. Webcam, image classification, and pose detection with Teachable Machine models — one short script away.

## Install

```
pip install kc-lib
```

Python 3.10–3.12 supported. Designed to install cleanly in a fresh PyCharm project.

## Quick start

```python
import kc

camera = kc.Camera()
camera.show()

model = kc.ImageModel("my_model.tflite")

for result in kc.predict_stream(camera, model):
    if result.is_confident:
        print(result.label)
```

Export your model from [Teachable Machine](https://teachablemachine.withgoogle.com) — choose **TensorFlow Lite → Floating point** when downloading.

## Status

Pre-alpha. API surface is settling; expect changes.
