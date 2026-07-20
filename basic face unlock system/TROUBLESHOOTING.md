# Troubleshooting Guide

## Installation Issues

### dlib Installation Problems

**Problem**: `pip install face_recognition` fails with compilation errors

**Solutions**:
1. **Use conda (recommended)**:
   \`\`\`bash
   conda install -c conda-forge dlib face_recognition
   \`\`\`

2. **Install system dependencies** (Ubuntu/Debian):
   \`\`\`bash
   sudo apt-get update
   sudo apt-get install build-essential cmake libopenblas-dev liblapack-dev libx11-dev libgtk-3-dev
   \`\`\`

3. **Windows**: Install Visual Studio Build Tools
4. **macOS**: Install Xcode command line tools:
   \`\`\`bash
   xcode-select --install
   \`\`\`

### OpenCV Issues

**Problem**: Camera not working or OpenCV import errors

**Solutions**:
1. Install OpenCV with proper backend:
   \`\`\`bash
   pip uninstall opencv-python
   pip install opencv-python-headless  # For servers
   # OR
   pip install opencv-python  # For desktop with GUI
   \`\`\`

2. Check camera permissions (macOS/Linux)
3. Try different camera index (0, 1, 2, etc.)

## Runtime Issues

### Camera Problems

**Problem**: "Camera not available" error

**Diagnosis**:
\`\`\`bash
python face_unlock.py test-camera --camera 0
\`\`\`

**Solutions**:
1. **Check camera permissions**:
   - macOS: System Preferences → Security & Privacy → Camera
   - Linux: Check if user is in `video` group
   
2. **Try different camera index**:
   \`\`\`bash
   python face_unlock.py test-camera --camera 1
   \`\`\`

3. **Close other applications** using the camera

4. **Check camera device**:
   \`\`\`bash
   # Linux
   ls /dev/video*
   
   # macOS
   system_profiler SPCameraDataType
   \`\`\`

### Face Detection Issues

**Problem**: "No face detected" during enrollment

**Solutions**:
1. **Improve lighting**: Use bright, even lighting
2. **Position face properly**: Center face in camera view
3. **Remove obstructions**: Take off glasses, hats, masks
4. **Check camera quality**: Ensure camera is clean and focused
5. **Try different detection model**:
   \`\`\`python
   # In face_core.py, change model parameter
   FaceCapture(model="cnn")  # If you have CUDA
   \`\`\`

### Encryption/Decryption Errors

**Problem**: "Decryption failed" or "Incorrect passphrase"

**Diagnosis**:
1. **Verify file integrity**:
   \`\`\`bash
   ls -la face_data/
   file face_data/*.enc
   \`\`\`

2. **Check file permissions**:
   \`\`\`bash
   ls -la face_data/*.enc
   # Should show: -rw------- (600 permissions)
   \`\`\`

**Solutions**:
1. **Double-check passphrase**: Ensure correct typing
2. **File corruption**: Re-enroll if file is corrupted
3. **Encoding issues**: Ensure consistent character encoding

### Performance Issues

**Problem**: Slow face detection or high CPU usage

**Solutions**:
1. **Use HOG model** (default, CPU-optimized):
   \`\`\`python
   FaceCapture(model="hog")
   \`\`\`

2. **Reduce frame size**:
   \`\`\`python
   FaceCapture(resize_factor=0.3)  # Smaller = faster
   \`\`\`

3. **Lower camera resolution** in face_core.py:
   \`\`\`python
   cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
   cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
   \`\`\`

4. **Close unnecessary applications**

## Verification Issues

### False Rejections

**Problem**: System rejects valid user

**Solutions**:
1. **Increase threshold**:
   \`\`\`bash
   python face_unlock.py verify --name user --threshold 0.65
   \`\`\`

2. **Re-enroll with more samples**:
   \`\`\`bash
   python face_unlock.py enroll --name user --count 15
   \`\`\`

3. **Improve enrollment conditions**: Better lighting, multiple angles

4. **Check for changes**: Glasses, facial hair, lighting differences

### False Acceptances

**Problem**: System accepts wrong person

**Solutions**:
1. **Decrease threshold**:
   \`\`\`bash
   python face_unlock.py verify --name user --threshold 0.45
   \`\`\`

2. **Enable liveness detection**:
   \`\`\`bash
   python face_unlock.py verify --name user --require-liveness
   \`\`\`

3. **Re-enroll with better quality samples**

### Liveness Detection Issues

**Problem**: Liveness detection always fails

**Solutions**:
1. **Check eye visibility**: Ensure eyes are clearly visible
2. **Blink naturally**: Don't force or exaggerate blinks
3. **Adjust parameters** in liveness.py:
   \`\`\`python
   LivenessDetector(ear_threshold=0.20)  # Lower = more sensitive
   \`\`\`

4. **Test liveness separately**:
   \`\`\`bash
   python scripts/test_liveness.py
   \`\`\`

## System Integration Issues

### Callback Script Problems

**Problem**: Callback script not executing

**Diagnosis**:
\`\`\`bash
# Test callback manually
./scripts/unlock_hook_example.sh test_user
echo $?  # Should return 0
\`\`\`

**Solutions**:
1. **Check script permissions**:
   \`\`\`bash
   chmod +x scripts/unlock_hook_example.sh
   \`\`\`

2. **Use absolute paths**:
   \`\`\`bash
   python face_unlock.py verify --name user --callback "/full/path/to/script.sh"
   \`\`\`

3. **Check script syntax**:
   \`\`\`bash
   bash -n scripts/unlock_hook_example.sh
   \`\`\`

### Environment Issues

**Problem**: Import errors or module not found

**Solutions**:
1. **Activate conda environment**:
   \`\`\`bash
   conda activate faceunlock
   \`\`\`

2. **Check Python path**:
   \`\`\`python
   import sys
   print(sys.path)
   \`\`\`

3. **Install in development mode**:
   \`\`\`bash
   pip install -e .
   \`\`\`

## Debugging Tips

### Enable Debug Mode

\`\`\`bash
python face_unlock.py verify --name user --debug
\`\`\`

### Check Logs

\`\`\`bash
tail -f /tmp/face_unlock.log
\`\`\`

### Test Individual Components

1. **Test camera**:
   \`\`\`bash
   python face_unlock.py test-camera
   \`\`\`

2. **Test liveness**:
   \`\`\`bash
   python scripts/test_liveness.py
   \`\`\`

3. **Test encryption**:
   \`\`\`bash
   python -m pytest tests/test_storage.py -v
   \`\`\`

### Common Error Messages

| Error | Likely Cause | Solution |
|-------|--------------|----------|
| "Camera not available" | Camera in use or no permissions | Close other apps, check permissions |
| "No face detected" | Poor lighting or positioning | Improve lighting, center face |
| "Decryption failed" | Wrong passphrase | Double-check passphrase |
| "Import error: face_recognition" | dlib not installed | Use conda installation |
| "Liveness check failed" | Eyes not detected | Ensure eyes are visible, blink naturally |

## Getting Help

1. **Check existing issues**: Look for similar problems
2. **Provide system info**: OS, Python version, camera type
3. **Include error messages**: Full error output
4. **Test with minimal setup**: Isolate the problem

## Performance Benchmarks

Expected performance on mid-range laptop:
- **Enrollment**: 5-10 seconds for 10 frames
- **Verification**: 5-10 FPS real-time
- **Memory usage**: ~200MB during operation
- **Storage**: ~1KB per enrollment (encrypted)

If performance is significantly worse, check system resources and camera quality.
