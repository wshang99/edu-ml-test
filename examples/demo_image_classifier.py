"""Demo: classify webcam frames with a Teachable Machine image model.

Setup:
  1. On https://teachablemachine.withgoogle.com, train an Image Project.
  2. Click "Export Model" -> TensorFlow Lite -> Floating point -> Download.
  3. Unzip the download. You'll get a model.tflite and a labels.txt.
  4. Put both files in the same folder as this script.

Run:
  python demo_image_classifier.py

Press Q in the preview window to quit.
"""

import kc

camera = kc.Camera()
camera.show()

model = kc.ImageModel("model.tflite")

for result in kc.predict_stream(camera, model):
    if result.is_confident:
        print(result)
