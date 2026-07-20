# Face Unlock Prototype - Enroll & Verify

A local Python prototype for face-based authentication using webcam enrollment and verification. This system captures face embeddings, encrypts them securely, and provides real-time face verification with optional liveness detection.

⚠️ **SECURITY WARNING**: This is a prototype for experimentation only. Not production-ready for security-critical applications. Face embeddings are sensitive biometric data - handle with care.

## Features

- **Enrollment Mode**: Capture multiple face frames, create averaged embedding, encrypt and store
- **Verification Mode**: Real-time face verification with configurable threshold
- **Encryption**: AES-GCM with PBKDF2 key derivation (100k iterations)
- **Liveness Detection**: Optional blink detection to prevent photo attacks
- **Integration Hooks**: Callback system for custom unlock actions
- **CPU Optimized**: Uses HOG face detection for performance

## Installation

### Recommended: Conda (avoids dlib build issues)

\`\`\`bash
# Clone repository
git clone <repo-url>
cd face-unlock-prototype

# Run setup script
chmod +x setup_conda.sh
./setup_conda.sh

# Activate environment
conda activate faceunlock
\`\`\`

### Alternative: pip

\`\`\`bash
# Install system dependencies first (Ubuntu/Debian)
sudo apt-get install cmake libopenblas-dev liblapack-dev libx11-dev libgtk-3-dev

# Install Python dependencies
pip install -r requirements.txt
\`\`\`

## Usage

### Enrollment

Capture and store encrypted face embedding:

\`\`\`bash
python face_unlock.py enroll --name me --count 12
# Enter passphrase when prompted (hidden input)
# Follow on-screen instructions in camera window
\`\`\`

### Verification

Verify identity against stored embedding:

\`\`\`bash
python face_unlock.py verify --name me --threshold 0.52 --require-liveness --callback "./unlock_hook.sh"
# Enter passphrase when prompted
# Look at camera for verification
\`\`\`

### CLI Options

**Enrollment:**
- `--name`: User identifier
- `--count`: Number of frames to capture (default: 10)
- `--out`: Output directory (default: face_data/)

**Verification:**
- `--name`: User to verify against
- `--threshold`: Distance threshold (default: 0.55, lower = stricter)
- `--require-liveness`: Enable blink detection
- `--callback`: Shell command to run on authorization

## Configuration & Tuning

### Threshold Values
- **0.45-0.50**: Very strict (may reject valid user)
- **0.55**: Default balanced setting
- **0.60-0.65**: More permissive (higher false accept risk)

### Capture Count
- **5-8**: Fast enrollment, less robust
- **10-12**: Recommended balance
- **15-20**: More robust, slower enrollment

### Liveness Detection
Enable with `--require-liveness` to prevent photo attacks. Requires at least one blink in 3-second window.

## Integration

### Callback Hook Example

Create `unlock_hook.sh`:
\`\`\`bash
#!/bin/bash
echo "$(date): Face unlock authorized for $1" >> /var/log/face_unlock.log
# Add your unlock logic here
# notify-send "Face Unlock" "Access granted"
\`\`\`

### Linux PAM Integration
For production use, consider integrating with:
- **Howdy**: Existing PAM module for face authentication
- **Custom PAM module**: Use this prototype as reference

**Do not attempt to bypass OS lock screens directly.**

## Security Notes

- Face embeddings are encrypted with AES-GCM
- Passphrases use PBKDF2 with 100k iterations
- No raw images stored by default
- Rate limiting prevents authorization flooding
- **GDPR Compliance**: Obtain user consent, provide deletion mechanism
- **Backup Security**: Encrypt backups, secure passphrase storage

## Testing

Run unit tests:
\`\`\`bash
pytest tests/ -v
\`\`\`

## File Structure

\`\`\`
face-unlock-prototype/
├── README.md
├── requirements.txt
├── setup_conda.sh
├── face_unlock.py          # Main CLI application
├── lib/
│   ├── storage.py          # Encryption/decryption
│   ├── face_core.py        # Face detection/recognition
│   ├── liveness.py         # Blink detection
│   └── utils.py            # Utilities
├── tests/
│   ├── test_storage.py
│   └── test_matching.py
└── face_data/              # Encrypted embeddings (.gitignored)
\`\`\`

## Troubleshooting

### Camera Issues
- Check camera permissions
- Ensure no other applications using camera
- Try different camera index if multiple cameras

### dlib Installation
- Use conda installation method
- On Windows: Install Visual Studio Build Tools
- On macOS: Install Xcode command line tools

### Performance
- Use `model="hog"` for CPU (default)
- Use `model="cnn"` only with CUDA-enabled dlib
- Reduce frame size for better performance

## License

MIT License - See LICENSE file for details.

## Disclaimer

This prototype is for educational and experimental purposes. Not suitable for production security systems. Users are responsible for compliance with local privacy and biometric data regulations.
