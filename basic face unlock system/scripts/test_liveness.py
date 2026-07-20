#!/usr/bin/env python3
"""
Face Unlock Prototype - Liveness Detection Test Script
MIT License

Interactive test script for liveness detection functionality.
"""

import sys
import os

# Add lib directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'lib'))

from lib.liveness import test_liveness_detection

if __name__ == "__main__":
    print("Face Unlock Prototype - Liveness Detection Test")
    print("=" * 50)
    print("This script will test the liveness detection system.")
    print("Look at the camera and blink naturally.")
    print("Press 'q' in the camera window to quit.")
    print()
    
    try:
        test_liveness_detection(camera_index=0, debug=True)
    except KeyboardInterrupt:
        print("\nTest cancelled by user.")
    except Exception as e:
        print(f"Test failed: {e}")
