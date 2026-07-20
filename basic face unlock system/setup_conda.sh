#!/bin/bash
# Face Unlock Prototype - Conda Environment Setup
# Recommended installation method to avoid dlib build issues

echo "Creating conda environment for face unlock prototype..."
conda create -n faceunlock python=3.10 -y
echo "Activating environment..."
conda activate faceunlock

echo "Installing core dependencies via conda-forge..."
conda install -c conda-forge dlib face_recognition opencv numpy pycryptodome pytest scikit-learn -y

echo "Installing additional requirements..."
pip install -r requirements.txt

echo "Setup complete! Activate with: conda activate faceunlock"
echo "Test installation with: python -c 'import face_recognition, cv2; print(\"All imports successful\")'"
