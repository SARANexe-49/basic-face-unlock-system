"""
Face Unlock Prototype - Utility Functions
MIT License

Common utilities and helper functions.
"""

import os
import sys
import time
from datetime import datetime
from typing import Optional


def get_timestamp() -> str:
    """Get current timestamp in ISO format."""
    return datetime.now().isoformat()


def ensure_directory(path: str) -> None:
    """Ensure directory exists, create if not."""
    os.makedirs(path, exist_ok=True)


def print_progress(current: int, total: int, message: str = "") -> None:
    """Print progress bar to console."""
    percent = (current / total) * 100
    bar_length = 30
    filled_length = int(bar_length * current // total)
    bar = '█' * filled_length + '-' * (bar_length - filled_length)
    
    print(f'\r[{bar}] {percent:.1f}% {message}', end='', flush=True)
    if current == total:
        print()  # New line when complete


def check_camera_available(camera_index: int = 0) -> bool:
    """Check if camera is available."""
    try:
        import cv2
        cap = cv2.VideoCapture(camera_index)
        if cap.isOpened():
            ret, _ = cap.read()
            cap.release()
            return ret
        return False
    except Exception:
        return False


def validate_name(name: str) -> bool:
    """Validate user name for file system safety."""
    if not name or len(name) < 1:
        return False
    
    # Check for invalid characters
    invalid_chars = '<>:"/\\|?*'
    return not any(char in name for char in invalid_chars)


class RateLimiter:
    """Simple rate limiter for authorization attempts."""
    
    def __init__(self, cooldown_seconds: float = 3.0):
        self.cooldown_seconds = cooldown_seconds
        self.last_success_time = 0.0
    
    def can_authorize(self) -> bool:
        """Check if enough time has passed since last authorization."""
        current_time = time.time()
        return (current_time - self.last_success_time) >= self.cooldown_seconds
    
    def record_success(self) -> None:
        """Record successful authorization timestamp."""
        self.last_success_time = time.time()
    
    def time_remaining(self) -> float:
        """Get remaining cooldown time in seconds."""
        current_time = time.time()
        elapsed = current_time - self.last_success_time
        return max(0.0, self.cooldown_seconds - elapsed)


def log_event(message: str, log_file: Optional[str] = None) -> None:
    """Log event with timestamp."""
    timestamp = get_timestamp()
    log_message = f"[{timestamp}] {message}"
    
    print(log_message)
    
    if log_file:
        try:
            with open(log_file, 'a') as f:
                f.write(log_message + '\n')
        except Exception as e:
            print(f"Warning: Could not write to log file {log_file}: {e}")


def safe_input(prompt: str, hidden: bool = False) -> str:
    """Safe input with optional hidden mode."""
    try:
        if hidden:
            import getpass
            return getpass.getpass(prompt)
        else:
            return input(prompt).strip()
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
        sys.exit(1)
    except EOFError:
        print("\nUnexpected end of input.")
        sys.exit(1)
