# Trabajo No. 2 - YoLo for Low-data Face Recognition

This directory is a self-contained runtime bundle for the YOLO identity detector and the secure ONNX web app.

## What is included

- `local_test/app_detector.py`: local Tkinter webcam detector
- `web_app_onnx/`: Flask server plus ONNX front-end
- `model/best.pt`: PyTorch checkpoint for the local detector
- `web_app_onnx/model/best.onnx`: ONNX model for the browser app
- `ResultsReport.pdf`: Architecture, data pipeline and results report
- `colab/photo_test.ipynb`: Colab notebook for uploading a photo and testing inference
- `bootstrap.sh`: installs dependencies and pulls YOLOv5 into `external/yolov5`

## Local runtime

Prerequisites:

- Python 3.10+
- `git`
- a working webcam for the Tkinter app

Setup:

```bash
cd delivery
./bootstrap.sh
source .venv/bin/activate
python local_test/app_detector.py
```

The local detector looks for:

- `delivery/model/best.pt`
- `delivery/external/yolov5`

If `external/yolov5` is missing, run `./bootstrap.sh` again.

## Web server

Run directly on the root of the repo:

```bash
source .venv/bin/activate
cd web_app_onnx
python app.py
```

The server listens on `http://localhost:5000`.

## Docker

Build and run the web server with Compose on the root of the repo:

```bash
docker compose up --build
```

The web app is available at `http://localhost:5000`.

To override the token used by the browser app:

```bash
API_TOKEN="your-token" docker compose up --build
```

## Colab photo test

Open `colab/photo_test.ipynb` in Google Colab, run the setup cell, then upload:

- a test image such as `photo.jpg`
- either `best.pt` or `best.onnx`

The notebook runs YOLOv5 detection and shows the annotated result inline.

Also available at [here](https://colab.research.google.com/drive/1WXoOpR96vgygj6mqsBlQf8hB8-wnXmgY?usp=sharing)

## Test web deployed app

The app is available at `https://yolo.02labs.me/` the token is:

```
EWCPGfYmPP0Qlu/80cBJg3PEEPMAmJWdYxpmsLMyqngNvITGSsTV2X+weI9j1pyR
BRpCc6BnIXg1NInHhVINhXUWiqa4+SohE8VppvHUNPohr2yPlUdQws6AGB/6qhwe
xCYYx+ld/PTkdH2B9w1aTXQZMH48bCfjKE4U1wPPGdI=
```
