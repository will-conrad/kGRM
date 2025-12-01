# KGRM - kNN Gesture Recognition Model

An interactive, touchless calculator that uses computer vision and machine learning to interpret hand gestures.

![Status](https://img.shields.io/badge/Status-Active-brightgreen) ![Python](https://img.shields.io/badge/Python-3.12-blue) ![Platform](https://img.shields.io/badge/Platform-Win%20|%20Mac%20|%20Linux-lightgrey)

## Project Overview

This application uses the **MediaPipe Hands** solution combined with a **k-Nearest Neighbor (kNN)** model to perform basic arithmetic operations based on hand gestures (finger counting for numbers, specific poses for operators).

---

## Installation and Setup

**Python 3.12 is Required** due to compatibility requirements with the current version of MediaPipe.

### 1. Initial Setup (Clone & Environment)
Start by cloning the repository and creating a Python virtual environment.

```bash
# 1. Clone the repository
git clone [YOUR_REPO_URL]
cd [YOUR_REPO_NAME]

# 2. Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate  # Use .venv\Scripts\activate on Windows

# 3. Update pip
python -m pip install --upgrade pip

# 4. Install all dependencies (OpenCV, MediaPipe, scikit-learn, etc.)
pip install -r requirements.txt

# 5. Run the file!
python main.py