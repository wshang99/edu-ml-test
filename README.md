# kc-lib

Machine learning for Python beginners. Webcam, image classification, and pose detection with Teachable Machine models — one short script away.

## Install (pre-release, from GitHub)

```
pip install --upgrade pip
pip install git+https://github.com/wshang99/edu-ml-test.git
```

Requires Python 3.10–3.14. Designed to install cleanly in a fresh PyCharm project — create a new project, let PyCharm pick the Python interpreter (3.11+ recommended), then run the commands above in the built-in terminal.

**If install fails with "No matching distribution found for ai-edge-litert":** you're on Python 3.9 or older. Install Python 3.11 from python.org and create a fresh PyCharm project pointing at it.

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
