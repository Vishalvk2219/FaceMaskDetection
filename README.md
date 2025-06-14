# 🛡️ Real-Time Face Mask Detection

This project detects whether a person is wearing a face mask, not wearing one, or wearing it incorrectly using a MobileNetV3-based classifier and MediaPipe for real-time face detection. The interface is built using Gradio to support webcam, image, and video file inputs.

## 🚀 Features

- 🔍 Real-time face detection using **MediaPipe**
- 🤖 Mask classification using **MobileNetV3 Large**
- 🖼️ Support for:
  - Image upload
  - Live webcam detection
  - Video file analysis
- 💻 Gradio-based interface (interactive and browser-ready)

## 🧠 Model

The model is a fine-tuned [MobileNetV3-Large](https://pytorch.org/vision/stable/models/generated/torchvision.models.mobilenet_v3_large.html) with the final classification head adjusted for 3 classes:
- `Mask`
- `No Mask`
- `Mask_Weared_Incorrect`

Trained model file: `mobilenetv3_mask_model.pth`

## 🛠️ Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/face-mask-detection.git
   cd face-mask-detection
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download the model file**

   Place the `mobilenetv3_mask_model.pth` file in the project root directory.

## 🧪 Usage

Launch the Gradio app:

```bash
python app.py
```

The app provides 3 modes:
- 📷 **Image Upload**
- 📸 **Live Webcam Feed**
- 🎥 **Video File Processing**

## 📦 Requirements

- Python 3.7+
- torch
- torchvision
- opencv-python
- mediapipe
- gradio
- pillow
- numpy

Install with:
```bash
pip install torch torchvision opencv-python mediapipe gradio pillow numpy
```

Or simply run:
```bash
pip install -r requirements.txt
```

## 📁 Project Structure

```
├── app.py                      # Main application script (Gradio UI)
├── mobilenetv3_mask_model.pth  # Trained model file (add this)
├── README.md
├── requirements.txt
```

Made with ❤️ using PyTorch, MediaPipe, and Gradio.
