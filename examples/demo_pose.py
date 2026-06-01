"""Demo: real-time pose detection with the pretrained MoveNet model.

No training required — kc.PoseModel('movenet') uses Google's pretrained weights,
bundled inside kc-lib.

Run:
  python demo_pose.py

A preview window opens with a skeleton drawn over you. Each frame also
prints how confident MoveNet is overall and where your nose is.

Press Q in the preview window to quit.
"""

import kc

camera = kc.Camera()
camera.show()

model = kc.PoseModel("movenet")

for result in kc.predict_stream(camera, model):
    if result.is_confident:
        nose_x, nose_y, nose_conf = result.keypoints["nose"]
        print(f"avg conf {result.confidence:.0%}, nose at ({int(nose_x)}, {int(nose_y)}) [{nose_conf:.0%}]")
