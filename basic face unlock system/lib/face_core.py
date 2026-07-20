"""
Face Unlock Prototype - Core Face Recognition
MIT License

Core functionality for face detection, encoding, and matching.
"""

import cv2
import face_recognition
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
import time

from .utils import print_progress, log_event


class FaceCapture:
    """Handles face detection and encoding from camera frames."""
    
    def __init__(self, model: str = "hog", resize_factor: float = 0.5):
        """
        Initialize face capture system.
        
        Args:
            model: Face detection model ("hog" for CPU, "cnn" for GPU)
            resize_factor: Factor to resize frames for speed (0.5 = half size)
        """
        self.model = model
        self.resize_factor = resize_factor
        self.frame_count = 0
        
    def detect_and_encode_face(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Detect face in frame and return encoding.
        
        Args:
            frame: Input image frame
            
        Returns:
            Face encoding array or None if no single face found
        """
        # Resize frame for speed
        if self.resize_factor != 1.0:
            small_frame = cv2.resize(frame, (0, 0), fx=self.resize_factor, fy=self.resize_factor)
        else:
            small_frame = frame
        
        # Convert BGR to RGB (face_recognition expects RGB)
        rgb_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        
        # Find face locations
        face_locations = face_recognition.face_locations(rgb_frame, model=self.model)
        
        # We want exactly one face
        if len(face_locations) != 1:
            return None
        
        # Get face encoding
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
        
        if len(face_encodings) == 1:
            return face_encodings[0]
        
        return None
    
    def capture_enrollment_frames(self, num_frames: int = 10, camera_index: int = 0) -> List[np.ndarray]:
        """
        Capture multiple frames for enrollment with real-time feedback.
        
        Args:
            num_frames: Number of good frames to capture
            camera_index: Camera device index
            
        Returns:
            List of face encodings
        """
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open camera {camera_index}")
        
        # Set camera properties for better quality
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        encodings = []
        frames_processed = 0
        consecutive_failures = 0
        max_consecutive_failures = 30  # About 1 second at 30fps
        
        print(f"Starting enrollment capture for {num_frames} frames...")
        print("Position your face in the camera view. Press 'q' to quit.")
        
        try:
            while len(encodings) < num_frames:
                ret, frame = cap.read()
                if not ret:
                    print("Failed to read from camera")
                    break
                
                frames_processed += 1
                
                # Try to detect and encode face
                encoding = self.detect_and_encode_face(frame)
                
                # Create display frame
                display_frame = frame.copy()
                
                if encoding is not None:
                    encodings.append(encoding)
                    consecutive_failures = 0
                    
                    # Draw success indicator
                    cv2.rectangle(display_frame, (10, 10), (300, 80), (0, 255, 0), -1)
                    cv2.putText(display_frame, f"Captured: {len(encodings)}/{num_frames}", 
                               (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
                    cv2.putText(display_frame, "Face detected - Good!", 
                               (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
                    
                    # Brief pause to avoid capturing too similar frames
                    time.sleep(0.2)
                    
                else:
                    consecutive_failures += 1
                    
                    # Draw failure indicator
                    cv2.rectangle(display_frame, (10, 10), (350, 80), (0, 0, 255), -1)
                    cv2.putText(display_frame, f"Captured: {len(encodings)}/{num_frames}", 
                               (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    
                    if consecutive_failures < 10:
                        cv2.putText(display_frame, "Position face in view", 
                                   (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    else:
                        cv2.putText(display_frame, "No face detected - adjust lighting", 
                                   (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # Draw progress bar
                progress_width = int((len(encodings) / num_frames) * 300)
                cv2.rectangle(display_frame, (10, 90), (310, 110), (100, 100, 100), -1)
                cv2.rectangle(display_frame, (10, 90), (10 + progress_width, 110), (0, 255, 0), -1)
                
                # Show frame
                cv2.imshow('Face Enrollment', display_frame)
                
                # Check for quit
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("\nEnrollment cancelled by user")
                    break
                
                # Check for too many consecutive failures
                if consecutive_failures > max_consecutive_failures:
                    print(f"\nToo many consecutive failures ({consecutive_failures}). Check camera and lighting.")
                    break
                
                # Console progress update
                if frames_processed % 30 == 0:  # Every ~1 second
                    print_progress(len(encodings), num_frames, f"frames captured")
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
        
        if len(encodings) < num_frames:
            print(f"\nWarning: Only captured {len(encodings)}/{num_frames} frames")
        else:
            print(f"\nSuccessfully captured {len(encodings)} frames!")
        
        return encodings
    
    def capture_verification_frame(self, camera_index: int = 0) -> Tuple[Optional[np.ndarray], np.ndarray]:
        """
        Capture single frame for verification.
        
        Args:
            camera_index: Camera device index
            
        Returns:
            Tuple of (face_encoding or None, original_frame)
        """
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open camera {camera_index}")
        
        try:
            ret, frame = cap.read()
            if not ret:
                return None, np.array([])
            
            encoding = self.detect_and_encode_face(frame)
            return encoding, frame
        
        finally:
            cap.release()


class FaceEmbedding:
    """Handles face embedding operations and matching."""
    
    @staticmethod
    def average_embeddings(embeddings: List[np.ndarray]) -> np.ndarray:
        """
        Compute average embedding from multiple face encodings.
        
        Args:
            embeddings: List of face encoding arrays
            
        Returns:
            Averaged embedding array
        """
        if not embeddings:
            raise ValueError("Cannot average empty embedding list")
        
        # Stack embeddings and compute mean
        embedding_matrix = np.array(embeddings)
        averaged = np.mean(embedding_matrix, axis=0)
        
        return averaged
    
    @staticmethod
    def compute_distance(embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Compute Euclidean distance between two face embeddings.
        
        Args:
            embedding1: First face embedding
            embedding2: Second face embedding
            
        Returns:
            Euclidean distance as float
        """
        return float(np.linalg.norm(embedding1 - embedding2))
    
    @staticmethod
    def is_match(embedding1: np.ndarray, embedding2: np.ndarray, threshold: float = 0.55) -> bool:
        """
        Determine if two embeddings match within threshold.
        
        Args:
            embedding1: First face embedding
            embedding2: Second face embedding
            threshold: Distance threshold for match
            
        Returns:
            True if embeddings match (distance < threshold)
        """
        distance = FaceEmbedding.compute_distance(embedding1, embedding2)
        return distance < threshold
    
    @staticmethod
    def create_enrollment_data(name: str, embeddings: List[np.ndarray]) -> Dict[str, Any]:
        """
        Create enrollment data structure from embeddings.
        
        Args:
            name: User name
            embeddings: List of face encodings
            
        Returns:
            Dictionary with enrollment data
        """
        if not embeddings:
            raise ValueError("Cannot create enrollment data from empty embeddings")
        
        averaged_embedding = FaceEmbedding.average_embeddings(embeddings)
        
        from .utils import get_timestamp
        
        return {
            "name": name,
            "embedding": averaged_embedding.tolist(),
            "created": get_timestamp(),
            "num_samples": len(embeddings)
        }


class VerificationSession:
    """Manages real-time face verification session."""
    
    def __init__(self, stored_embedding: np.ndarray, threshold: float = 0.55, 
                 cooldown_seconds: float = 3.0):
        """
        Initialize verification session.
        
        Args:
            stored_embedding: Reference embedding to match against
            threshold: Distance threshold for authorization
            cooldown_seconds: Cooldown between successful authorizations
        """
        self.stored_embedding = stored_embedding
        self.threshold = threshold
        self.face_capture = FaceCapture()
        
        from .utils import RateLimiter
        self.rate_limiter = RateLimiter(cooldown_seconds)
        
        self.last_distance = None
        self.frame_count = 0
    
    def verify_frame(self, frame: np.ndarray) -> Tuple[bool, float, str]:
        """
        Verify face in single frame.
        
        Args:
            frame: Input camera frame
            
        Returns:
            Tuple of (authorized, distance, status_message)
        """
        self.frame_count += 1
        
        # Detect and encode face
        encoding = self.face_capture.detect_and_encode_face(frame)
        
        if encoding is None:
            return False, -1.0, "No face detected"
        
        # Compute distance
        distance = FaceEmbedding.compute_distance(self.stored_embedding, encoding)
        self.last_distance = distance
        
        # Check if within threshold
        if distance < self.threshold:
            # Check rate limiting
            if self.rate_limiter.can_authorize():
                self.rate_limiter.record_success()
                return True, distance, "AUTHORIZED"
            else:
                remaining = self.rate_limiter.time_remaining()
                return False, distance, f"Rate limited ({remaining:.1f}s remaining)"
        else:
            return False, distance, "Not authorized"
    
    def run_continuous_verification(self, camera_index: int = 0, 
                                  callback_command: Optional[str] = None,
                                  require_liveness: bool = False) -> None:
        """
        Run continuous verification loop with camera display.
        
        Args:
            camera_index: Camera device index
            callback_command: Shell command to run on authorization
            require_liveness: Whether to require liveness detection
        """
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open camera {camera_index}")
        
        # Set camera properties
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        print("Starting face verification...")
        print("Look at the camera. Press 'q' to quit.")
        
        # Initialize liveness detector if required
        liveness_detector = None
        if require_liveness:
            from .liveness import LivenessDetector
            liveness_detector = LivenessDetector()
            print("Liveness detection enabled - blink naturally")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Failed to read from camera")
                    break
                
                # Check liveness first if required
                liveness_passed = True
                if liveness_detector:
                    liveness_passed = liveness_detector.check_liveness(frame)
                
                # Verify face
                authorized, distance, status = self.verify_frame(frame)
                
                # Override authorization if liveness failed
                if authorized and not liveness_passed:
                    authorized = False
                    status = "Liveness check failed"
                
                # Create display frame
                display_frame = frame.copy()
                
                # Draw status overlay
                if authorized:
                    # Green background for authorized
                    cv2.rectangle(display_frame, (10, 10), (400, 100), (0, 255, 0), -1)
                    cv2.putText(display_frame, "AUTHORIZED", 
                               (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 3)
                    cv2.putText(display_frame, f"Distance: {distance:.3f}", 
                               (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
                    
                    # Execute callback
                    if callback_command:
                        self._execute_callback(callback_command)
                    
                    # Log authorization
                    log_event(f"AUTHORIZED: Distance {distance:.3f}")
                    
                else:
                    # Red background for not authorized
                    cv2.rectangle(display_frame, (10, 10), (400, 100), (0, 0, 255), -1)
                    cv2.putText(display_frame, "NOT AUTHORIZED", 
                               (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                    
                    if distance >= 0:
                        cv2.putText(display_frame, f"Distance: {distance:.3f} (>{self.threshold:.3f})", 
                                   (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    else:
                        cv2.putText(display_frame, status, 
                                   (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # Draw liveness status if enabled
                if liveness_detector:
                    liveness_status = "Liveness: PASS" if liveness_passed else "Liveness: FAIL"
                    color = (0, 255, 0) if liveness_passed else (0, 0, 255)
                    cv2.putText(display_frame, liveness_status, 
                               (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                
                # Show frame
                cv2.imshow('Face Verification', display_frame)
                
                # Check for quit
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("\nVerification stopped by user")
                    break
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
    
    def _execute_callback(self, command: str) -> None:
        """Execute callback command safely."""
        try:
            import subprocess
            result = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                log_event(f"Callback executed successfully: {command}")
            else:
                log_event(f"Callback failed (code {result.returncode}): {command}")
        except subprocess.TimeoutExpired:
            log_event(f"Callback timed out: {command}")
        except Exception as e:
            log_event(f"Callback error: {command} - {e}")
