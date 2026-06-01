# kc-lib pre-release teacher test

Thanks for trying this. It's a Python library for classroom ML — modeled on machinelearningforkids.co.uk but Python-native. It's pre-alpha and the whole point of this round is to find install friction and confirm the basic flow works on machines that aren't the developer's.

About 20 minutes of your time, including training a tiny model.

---

## What you'll need

- A laptop with a webcam (Windows, macOS, or Linux are all fine)
- **Python 3.10 or newer** (3.11 recommended). If you're on 3.9 or older, see Step 1.
- **PyCharm Community Edition** (free) — https://www.jetbrains.com/pycharm/download/

## Step 1: Get Python 3.11 (skip if you already have 3.10+)

To check: open a terminal/PowerShell and run `python --version`.

If it says 3.9 or older (or you get "not found"):
1. Download Python 3.11 from https://www.python.org/downloads/
2. Run the installer.
3. **Important:** tick "Add Python to PATH" on the first screen.
4. Install.

## Step 2: Create a fresh PyCharm project

1. Open PyCharm → **File → New Project**.
2. Name: `kc-test`. Location: anywhere.
3. Interpreter type: **"Project venv"** (the default). Python version: pick **3.11** (or 3.10, 3.12, 3.13, or 3.14 — any of those work).
4. Click **Create**.
5. Open the built-in terminal: **View → Tool Windows → Terminal** (or `Alt + F12`).

## Step 3: Install the library

In the PyCharm terminal, paste these two lines:

```
pip install --upgrade pip
pip install git+https://github.com/wshang99/edu-ml-test.git
```

Takes 1–3 minutes. About 75 MB of downloads.

## Step 4: Confirm it imported

Still in the PyCharm terminal:

```
python -c "import kc; print('kc version:', kc.__version__)"
```

Expected: `kc version: 0.0.1`

## Step 5: Train a tiny model on Teachable Machine

1. Go to https://teachablemachine.withgoogle.com → **Get Started** → **Image Project** → **Standard image model**.
2. Make 2 or 3 classes (e.g. "thumbs up", "thumbs down", "neutral"). Rename the classes by clicking on "Class 1" etc.
3. For each class, click **Webcam** and hold your gesture for ~5 seconds — it captures about 50 frames automatically.
4. Click **Train Model**. Takes ~30 seconds.
5. Click **Export Model** → **TensorFlow Lite** tab → select **Floating point** → click **Download my model**.
6. Unzip the file you downloaded. You'll see `model.tflite` and `labels.txt` (plus other files — ignore them).

## Step 6: Drop the model into the project + run the demo

1. In your file manager, copy `model.tflite` and `labels.txt` into the **kc-test** project folder (the folder PyCharm opened).
2. In PyCharm, right-click the project folder → **New → Python File** → name it `demo`. Paste this:

```python
import kc

camera = kc.Camera()
camera.show()

model = kc.ImageModel("model.tflite")

for result in kc.predict_stream(camera, model):
    if result.is_confident:
        print(result)
```

3. Right-click anywhere in the editor → **Run 'demo'**.
4. A window should open showing your webcam feed. As you do the gestures you trained, the top-left of the video labels what it sees, and the terminal prints the same.
5. Press **Q** in the video window to close cleanly.

---

## What to send back

Even a thumbs-up is useful. If anything was friction:

- **OS + Python version.** Run `python --version` in the PyCharm terminal.
- **Which step number** you got stuck on.
- **The exact error message** if there was one (copy the last ~15 lines from the terminal).
- **Anything that felt confusing** — even if you got past it, that's signal.

## If something fails — common ones

| Symptom | Likely cause | Fix |
|---|---|---|
| `pip install` says "Package 'kc-lib' requires a different Python" | You're on Python 3.9 or older | Do Step 1 and create a new project on 3.11 |
| `pip install` says "No matching distribution found for ai-edge-litert" | Same as above (Python too old) | Same |
| `pip install` says "ERROR: Could not find a version that satisfies the requirement..." after `pip install --upgrade pip` | Sometimes a stale cache. | `pip install --no-cache-dir git+https://github.com/wshang99/edu-ml-test.git` |
| Demo window opens but stays black | Another app (Zoom, Teams, browser) is holding the webcam | Close those apps, re-run |
| Demo runs but predictions are wildly off | `labels.txt` isn't beside `model.tflite` | Put both files in the same folder as `demo.py` |
| Q key doesn't close the window | Click the X button on the video window instead | Avoid `Ctrl+C` in the terminal — it leaves the camera in a weird state |

## Bonus: try the pretrained pose detector (no training needed)

Once Step 4 passes, you can also try the pretrained MoveNet pose detector — no Teachable Machine setup required. Create a second file `demo_pose.py` in the same project folder:

```python
import kc

camera = kc.Camera()
camera.show()
model = kc.PoseModel("movenet")

for result in kc.predict_stream(camera, model):
    if result.is_confident:
        nose_x, nose_y, _ = result.keypoints["nose"]
        print(f"nose at ({int(nose_x)}, {int(nose_y)})")
```

Run it. You should see a skeleton drawn over yourself on the preview window. Pose models can run alongside image models — no extra install steps.

## What's intentionally not working yet

- `kc.ImageModel("mobilenet")` — pretrained ImageNet classifier, not wired in this version
- `kc.PoseModel("<path-to-TM-pose-export>.tflite")` — Teachable Machine Pose Projects load but `predict()` is stubbed. Use `kc.PoseModel("movenet")` for pretrained pose instead.

If you hit one of those, you'll see a `NotImplementedError` with a "not wired yet" message. Expected, not a bug.
