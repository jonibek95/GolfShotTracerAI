# ⛳ Golf Ball Trajectory Visualization

This repository demonstrates how to detect a **golf ball**, estimate its **flight parameters**, and **visualize the 3D trajectory** directly onto the video.

---

## 🎥 Result Video

![Golf_with_trajectory](https://github.com/jonibek95/Golf_Trajectory/assets/84657258/5946c79e-2728-44a0-b856-7727064357a9)

---

## 🔍 Overview

This project detects a golf ball using a YOLO-based model, calculates its initial velocity and angles, and simulates the 3D physical trajectory of the ball using motion equations and physics-based spin models. The result is rendered as a dynamic colored arc over the original video.

---

## 🧠 Key Components

- **YOLOv5** for detecting the golf ball's position
- **Trajectory simulation** using projectile motion + spin dynamics
- **3D rotation and projection** to align the trajectory with the camera view
- **Smooth spline interpolation** for clean rendering
- **OpenCV** to draw trajectory frames and export the final video

---

## 🗂️ File Structure

```
Golf_Trajectory/
├── classify/               # Custom classification utilities
├── data/                   # Input video files
├── models/                 # YOLOv5 model (e.g., best.pt)
├── utils/                  # Math, drawing, and support functions
├── Draw_trajectory.ipynb   # Main notebook to run the project
├── mathcalc.py             # Physics and motion calculation logic
├── detect.py               # YOLO detection wrapper
├── golf_with_trajectory.mp4
├── Golf_with_trajectory.gif
└── ...
```

---

## 🚀 How to Run

### 1. Clone the Repository

```bash
git clone https://github.com/jonibek95/Golf_Trajectory.git
cd Golf_Trajectory
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

Make sure you have:
- Python 3.8+
- OpenCV, NumPy, SciPy
- PyTorch and YOLOv5-compatible weights

### 3. Run the Notebook or Script

- To visualize inside Jupyter:
```bash
jupyter notebook Draw_trajectory.ipynb
```

---

## ⚙️ Configuration

You can adjust:
- `bspeed`: ball speed
- `langle`: launch angle
- `spinr`: side spin rate
- `sprate`: spin rate
- `sangle`: spin axis angle

These parameters affect the shape, height, and curve of the simulated flight path.

---

## 📊 Output

- `golf_with_trajectory.mp4`: Final output video with dynamic trajectory
- `Golf_with_trajectory.gif`: Sample for demo purposes
- Colored arc that changes as the ball flies

---

## 🧪 Physics Engine

Physics is handled with:
- Drag
- Magnus effect (from spin)
- Gravity
- Angle-based projection

Trajectory points are rotated and projected back to image coordinates and visualized frame-by-frame.
