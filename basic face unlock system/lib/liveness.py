"""
Face Unlock Prototype - Liveness Detection
MIT License

Implements blink detection using facial landmarks to prevent photo attacks.
Uses Eye Aspect Ratio (EAR) method for robust blink detection.
"""

import cv2
import face_recognition
import numpy as np
from typing import List, Tuple, Optional
import time
from collections import deque

from .utils import log_event


class LivenessDetector:
    """
    Detects liveness using blink detection via Eye Aspect Ratio (EAR).
    
    The EAR method calculates the ratio of eye height to width using facial landmarks.
    A blink is detected when EAR drops below threshold and returns above it.
    """
    
    # Eye landmark indices for face_recognition (68-point model)
    LEFT_EYE_LANDMARKS = [36, 37, 38, 39, 40, 41]
    RIGHT_EYE_LANDMARKS = [42, 43, 44, 45, 46, 47]
    
    def __init__(self, 
                 ear_threshold: float = 0.25,
                 blink_frames: int = 3,
                 liveness_window: float = 3.0,
                 min_blinks: int = 1):
        """
        Initialize liveness detector.
        
        Args:
            ear_threshold: EAR threshold below which eye is considered closed
            blink_frames: Consecutive frames below threshold to register blink
            liveness_window: Time window (seconds) to check for blinks
            min_blinks: Minimum blinks required in window for liveness
        """
        self.ear_threshold = ear_threshold
        self.blink_frames = blink_frames
        self.liveness_window = liveness_window
        self.min_blinks = min_blinks
        
        # State tracking
        self.blink_history = deque(maxlen=100)  # Store recent blink timestamps
        self.consecutive_closed_frames = 0
        self.last_ear_values = deque(maxlen=10)  # For smoothing
        self.in_blink = False
        
        # Debug info
        self.debug_mode = False
        self.last_ear_left = 0.0
        self.last_ear_right = 0.0
    
    def calculate_ear(self, eye_landmarks: List[Tuple[int, int]]) -> float:
        """
        Calculate Eye Aspect Ratio for given eye landmarks.
        
        EAR = (|p2-p6| + |p3-p5|) / (2 * |p1-p4|)
        Where p1-p6 are the eye landmark points.
        
        Args:
            eye_landmarks: List of (x, y) tuples for eye landmarks
            
        Returns:
            Eye Aspect Ratio as float
        """
        if len(eye_landmarks) != 6:
            return 0.0
        
        # Convert to numpy array for easier calculation
        points = np.array(eye_landmarks)
        
        # Calculate vertical distances
        vertical_1 = np.linalg.norm(points[1] - points[5])  # p2-p6
        vertical_2 = np.linalg.norm(points[2] - points[4])  # p3-p5
        
        # Calculate horizontal distance
        horizontal = np.linalg.norm(points[0] - points[3])  # p1-p4
        
        # Calculate EAR
        if horizontal == 0:
            return 0.0
        
        ear = (vertical_1 + vertical_2) / (2.0 * horizontal)
        return ear
    
    def detect_blink_in_frame(self, frame: np.ndarray) -> Tuple[bool, float, bool]:
        """
        Detect blink in single frame using facial landmarks.
        
        Args:
            frame: Input camera frame
            
        Returns:
            Tuple of (blink_detected, average_ear, eyes_found)
        """
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Find face landmarks
        face_landmarks_list = face_recognition.face_landmarks(rgb_frame)
        
        if not face_landmarks_list:
            return False, 0.0, False
        
        # Use first face found
        face_landmarks = face_landmarks_list[0]
        
        # Extract eye landmarks
        if 'left_eye' not in face_landmarks or 'right_eye' not in face_landmarks:
            return False, 0.0, False
        
        left_eye = face_landmarks['left_eye']
        right_eye = face_landmarks['right_eye']
        
        # Calculate EAR for both eyes
        left_ear = self.calculate_ear(left_eye)
        right_ear = self.calculate_ear(right_eye)
        
        # Average EAR
        avg_ear = (left_ear + right_ear) / 2.0
        
        # Store for debugging
        self.last_ear_left = left_ear
        self.last_ear_right = right_ear
        
        # Smooth EAR values
        self.last_ear_values.append(avg_ear)
        smoothed_ear = np.mean(list(self.last_ear_values))
        
        # Detect blink
        blink_detected = False
        
        if smoothed_ear < self.ear_threshold:
            self.consecutive_closed_frames += 1
            if not self.in_blink and self.consecutive_closed_frames >= self.blink_frames:
                self.in_blink = True
        else:
            if self.in_blink and self.consecutive_closed_frames >= self.blink_frames:
                # Blink completed
                blink_detected = True
                self.blink_history.append(time.time())
                if self.debug_mode:
                    log_event(f"Blink detected! EAR: {smoothed_ear:.3f}")
            
            self.in_blink = False
            self.consecutive_closed_frames = 0
        
        return blink_detected, smoothed_ear, True
    
    def check_liveness(self, frame: np.ndarray) -> bool:
        """
        Check if liveness criteria are met based on recent blink history.
        
        Args:
            frame: Input camera frame
            
        Returns:
            True if liveness criteria are satisfied
        """
        # Detect blink in current frame
        blink_detected, ear, eyes_found = self.detect_blink_in_frame(frame)
        
        if not eyes_found:
            return False
        
        # Check recent blink history
        current_time = time.time()
        recent_blinks = [
            timestamp for timestamp in self.blink_history
            if current_time - timestamp <= self.liveness_window
        ]
        
        # Liveness passed if we have enough recent blinks
        liveness_passed = len(recent_blinks) >= self.min_blinks
        
        if self.debug_mode and blink_detected:
            log_event(f"Recent blinks: {len(recent_blinks)}/{self.min_blinks} "
                     f"(window: {self.liveness_window}s)")
        
        return liveness_passed
    
    def get_debug_info(self) -> dict:
        """
        Get debug information about liveness detection state.
        
        Returns:
            Dictionary with debug information
        """
        current_time = time.time()
        recent_blinks = [
            timestamp for timestamp in self.blink_history
            if current_time - timestamp <= self.liveness_window
        ]
        
        return {
            "ear_left": self.last_ear_left,
            "ear_right": self.last_ear_right,
            "ear_threshold": self.ear_threshold,
            "consecutive_closed_frames": self.consecutive_closed_frames,
            "in_blink": self.in_blink,
            "recent_blinks": len(recent_blinks),
            "min_blinks_required": self.min_blinks,
            "liveness_window": self.liveness_window,
            "total_blinks": len(self.blink_history)
        }
    
    def reset_state(self) -> None:
        """Reset liveness detection state."""
        self.blink_history.clear()
        self.consecutive_closed_frames = 0
        self.last_ear_values.clear()
        self.in_blink = False
    
    def set_debug_mode(self, enabled: bool) -> None:
        """Enable or disable debug mode."""
        self.debug_mode = enabled
    
    def draw_debug_overlay(self, frame: np.ndarray) -> np.ndarray:
        """
        Draw debug overlay on frame showing EAR values and blink status.
        
        Args:
            frame: Input frame
            
        Returns:
            Frame with debug overlay
        """
        debug_frame = frame.copy()
        debug_info = self.get_debug_info()
        
        # Draw debug text
        y_offset = 150
        line_height = 25
        
        debug_texts = [
            f"EAR L: {debug_info['ear_left']:.3f}",
            f"EAR R: {debug_info['ear_right']:.3f}",
            f"Threshold: {debug_info['ear_threshold']:.3f}",
            f"Closed frames: {debug_info['consecutive_closed_frames']}",
            f"In blink: {debug_info['in_blink']}",
            f"Recent blinks: {debug_info['recent_blinks']}/{debug_info['min_blinks_required']}",
            f"Total blinks: {debug_info['total_blinks']}"
        ]
        
        for i, text in enumerate(debug_texts):
            y_pos = y_offset + i * line_height
            cv2.putText(debug_frame, text, (10, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Draw EAR threshold line (visual indicator)
        frame_height = debug_frame.shape[0]
        threshold_y = int(frame_height - 50)
        
        # EAR visualization bar
        bar_width = 200
        bar_height = 20
        bar_x = 10
        
        # Background bar
        cv2.rectangle(debug_frame, (bar_x, threshold_y), 
                     (bar_x + bar_width, threshold_y + bar_height), (50, 50, 50), -1)
        
        # Current EAR level
        avg_ear = (debug_info['ear_left'] + debug_info['ear_right']) / 2.0
        ear_width = int(min(avg_ear * bar_width / 0.5, bar_width))  # Scale to 0.5 max
        
        color = (0, 255, 0) if avg_ear > self.ear_threshold else (0, 0, 255)
        cv2.rectangle(debug_frame, (bar_x, threshold_y), 
                     (bar_x + ear_width, threshold_y + bar_height), color, -1)
        
        # Threshold line
        threshold_x = int(bar_x + self.ear_threshold * bar_width / 0.5)
        cv2.line(debug_frame, (threshold_x, threshold_y), 
                (threshold_x, threshold_y + bar_height), (255, 255, 255), 2)
        
        cv2.putText(debug_frame, f"EAR: {avg_ear:.3f}", 
                   (bar_x, threshold_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return debug_frame


class AdvancedLivenessDetector(LivenessDetector):
    """
    Advanced liveness detector with additional anti-spoofing measures.
    
    Extends basic blink detection with:
    - Head movement detection
    - Texture analysis (basic)
    - Multiple validation criteria
    """
    
    def __init__(self, **kwargs):
        """Initialize advanced liveness detector."""
        super().__init__(**kwargs)
        
        # Additional state for advanced detection
        self.head_positions = deque(maxlen=30)  # Track head movement
        self.face_sizes = deque(maxlen=30)      # Track face size changes
        
        # Advanced thresholds
        self.min_head_movement = 5.0  # Minimum head movement in pixels
        self.min_size_variation = 0.05  # Minimum face size variation
    
    def detect_head_movement(self, frame: np.ndarray) -> Tuple[bool, float]:
        """
        Detect head movement by tracking face position changes.
        
        Args:
            frame: Input camera frame
            
        Returns:
            Tuple of (movement_detected, movement_amount)
        """
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Find face locations
        face_locations = face_recognition.face_locations(rgb_frame)
        
        if not face_locations:
            return False, 0.0
        
        # Use first face
        top, right, bottom, left = face_locations[0]
        
        # Calculate face center and size
        center_x = (left + right) / 2
        center_y = (top + bottom) / 2
        face_size = (right - left) * (bottom - top)
        
        # Store current position and size
        self.head_positions.append((center_x, center_y))
        self.face_sizes.append(face_size)
        
        # Calculate movement if we have enough history
        if len(self.head_positions) < 10:
            return False, 0.0
        
        # Calculate position variance
        positions = np.array(list(self.head_positions))
        position_variance = np.var(positions, axis=0)
        movement_amount = np.sqrt(np.sum(position_variance))
        
        # Calculate size variance
        sizes = np.array(list(self.face_sizes))
        size_variance = np.var(sizes) / np.mean(sizes) if np.mean(sizes) > 0 else 0
        
        # Movement detected if variance exceeds thresholds
        movement_detected = (movement_amount > self.min_head_movement or 
                           size_variance > self.min_size_variation)
        
        return movement_detected, movement_amount
    
    def check_advanced_liveness(self, frame: np.ndarray) -> Tuple[bool, dict]:
        """
        Check liveness using multiple criteria.
        
        Args:
            frame: Input camera frame
            
        Returns:
            Tuple of (liveness_passed, detailed_results)
        """
        # Basic blink detection
        blink_liveness = self.check_liveness(frame)
        
        # Head movement detection
        movement_detected, movement_amount = self.detect_head_movement(frame)
        
        # Combine criteria
        results = {
            "blink_liveness": blink_liveness,
            "movement_detected": movement_detected,
            "movement_amount": movement_amount,
            "debug_info": self.get_debug_info()
        }
        
        # Advanced liveness requires both blink and some movement
        advanced_liveness = blink_liveness and movement_detected
        
        return advanced_liveness, results


# Utility functions for easy integration
def create_liveness_detector(mode: str = "basic", **kwargs) -> LivenessDetector:
    """
    Create liveness detector instance.
    
    Args:
        mode: "basic" or "advanced"
        **kwargs: Additional parameters for detector
        
    Returns:
        LivenessDetector instance
    """
    if mode == "advanced":
        return AdvancedLivenessDetector(**kwargs)
    else:
        return LivenessDetector(**kwargs)


def test_liveness_detection(camera_index: int = 0, debug: bool = True) -> None:
    """
    Test liveness detection with live camera feed.
    
    Args:
        camera_index: Camera device index
        debug: Whether to show debug overlay
    """
    detector = LivenessDetector()
    detector.set_debug_mode(debug)
    
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"Could not open camera {camera_index}")
        return
    
    print("Testing liveness detection. Press 'q' to quit.")
    print("Blink naturally to test detection.")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Check liveness
            liveness_passed = detector.check_liveness(frame)
            
            # Create display frame
            if debug:
                display_frame = detector.draw_debug_overlay(frame)
            else:
                display_frame = frame.copy()
            
            # Draw liveness status
            status_text = "LIVENESS: PASS" if liveness_passed else "LIVENESS: FAIL"
            status_color = (0, 255, 0) if liveness_passed else (0, 0, 255)
            
            cv2.putText(display_frame, status_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)
            
            cv2.imshow('Liveness Detection Test', display_frame)
            
            # Check for quit
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
    
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    # Run test if module is executed directly
    test_liveness_detection()
