#!/usr/bin/env python3
"""
Face Unlock Prototype - Main CLI Application
MIT License

Command-line interface for face enrollment and verification.
Provides secure face-based authentication with optional liveness detection.

Usage:
    python face_unlock.py enroll --name <name> [options]
    python face_unlock.py verify --name <name> [options]
    python face_unlock.py list
    python face_unlock.py delete --name <name>
"""

import sys
import os
import argparse
import numpy as np
from typing import Optional

# Add lib directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'lib'))

from lib.face_core import FaceCapture, FaceEmbedding, VerificationSession
from lib.storage import SecureStorage, get_secure_passphrase, EncryptionError
from lib.liveness import create_liveness_detector
from lib.utils import (
    check_camera_available, validate_name, log_event, 
    get_timestamp, safe_input
)


class FaceUnlockCLI:
    """Main CLI application class."""
    
    def __init__(self):
        """Initialize CLI application."""
        self.storage = SecureStorage()
        self.face_capture = FaceCapture()
    
    def enroll_user(self, name: str, count: int = 10, camera_index: int = 0, 
                   storage_dir: str = "face_data") -> bool:
        """
        Enroll a new user with face capture.
        
        Args:
            name: User name
            count: Number of frames to capture
            camera_index: Camera device index
            storage_dir: Storage directory
            
        Returns:
            True if enrollment successful
        """
        try:
            # Validate inputs
            if not validate_name(name):
                print(f"Error: Invalid name '{name}'. Use alphanumeric characters only.")
                return False
            
            if count < 5 or count > 50:
                print(f"Error: Frame count must be between 5 and 50, got {count}")
                return False
            
            # Check if user already exists
            if self.storage.enrollment_exists(name):
                response = safe_input(f"User '{name}' already exists. Overwrite? (y/N): ")
                if response.lower() != 'y':
                    print("Enrollment cancelled.")
                    return False
            
            # Check camera availability
            if not check_camera_available(camera_index):
                print(f"Error: Camera {camera_index} not available or accessible.")
                return False
            
            print(f"Starting enrollment for user: {name}")
            print(f"Will capture {count} frames for face recognition.")
            print("Make sure you have good lighting and look directly at the camera.")
            
            # Get passphrase
            try:
                passphrase = get_secure_passphrase("Enter passphrase for encryption: ", confirm=True)
            except ValueError as e:
                print(f"Error: {e}")
                return False
            
            # Capture face frames
            print("\nStarting face capture...")
            encodings = self.face_capture.capture_enrollment_frames(count, camera_index)
            
            if len(encodings) < max(3, count // 2):
                print(f"Error: Insufficient face samples captured ({len(encodings)}/{count})")
                print("Try again with better lighting or camera positioning.")
                return False
            
            # Create enrollment data
            enrollment_data = FaceEmbedding.create_enrollment_data(name, encodings)
            
            # Save encrypted enrollment
            file_path = self.storage.save_enrollment(name, enrollment_data, passphrase)
            
            # Success message
            print(f"\n[ENROLL] Captured {len(encodings)}/{count} frames. Averaged embedding created.")
            print(f"Enrollment saved to {file_path} (encrypted).")
            print(f"Created: {enrollment_data['created']}")
            
            log_event(f"User '{name}' enrolled successfully with {len(encodings)} samples")
            
            return True
            
        except KeyboardInterrupt:
            print("\nEnrollment cancelled by user.")
            return False
        except Exception as e:
            print(f"Enrollment failed: {e}")
            return False
    
    def verify_user(self, name: str, threshold: float = 0.55, camera_index: int = 0,
                   callback_command: Optional[str] = None, require_liveness: bool = False,
                   debug: bool = False) -> bool:
        """
        Verify user identity against stored enrollment.
        
        Args:
            name: User name to verify
            threshold: Distance threshold for authorization
            camera_index: Camera device index
            callback_command: Command to run on authorization
            require_liveness: Whether to require liveness detection
            debug: Enable debug mode
            
        Returns:
            True if verification session completed successfully
        """
        try:
            # Validate inputs
            if not validate_name(name):
                print(f"Error: Invalid name '{name}'")
                return False
            
            if not self.storage.enrollment_exists(name):
                print(f"Error: No enrollment found for user '{name}'")
                print("Available users:", ", ".join(self.storage.list_enrollments()))
                return False
            
            if threshold < 0.1 or threshold > 1.0:
                print(f"Error: Threshold must be between 0.1 and 1.0, got {threshold}")
                return False
            
            # Check camera availability
            if not check_camera_available(camera_index):
                print(f"Error: Camera {camera_index} not available or accessible.")
                return False
            
            # Get passphrase
            try:
                passphrase = get_secure_passphrase("Enter passphrase: ")
            except ValueError as e:
                print(f"Error: {e}")
                return False
            
            # Load enrollment data
            try:
                enrollment_data = self.storage.load_enrollment(name, passphrase)
            except EncryptionError:
                print("Error: Incorrect passphrase or corrupted enrollment data.")
                return False
            
            # Extract stored embedding
            stored_embedding = np.array(enrollment_data['embedding'])
            
            print(f"Starting verification for user: {name}")
            print(f"Threshold: {threshold}")
            print(f"Liveness detection: {'enabled' if require_liveness else 'disabled'}")
            if callback_command:
                print(f"Callback command: {callback_command}")
            
            # Create verification session
            verification = VerificationSession(stored_embedding, threshold)
            
            # Run verification
            verification.run_continuous_verification(
                camera_index=camera_index,
                callback_command=callback_command,
                require_liveness=require_liveness
            )
            
            return True
            
        except KeyboardInterrupt:
            print("\nVerification stopped by user.")
            return True  # Not an error, user chose to quit
        except Exception as e:
            print(f"Verification failed: {e}")
            return False
    
    def list_users(self) -> bool:
        """
        List all enrolled users.
        
        Returns:
            True if successful
        """
        try:
            users = self.storage.list_enrollments()
            
            if not users:
                print("No enrolled users found.")
                return True
            
            print(f"Enrolled users ({len(users)}):")
            for user in users:
                info = self.storage.get_enrollment_info(user)
                if info:
                    size_kb = info['file_size'] / 1024
                    print(f"  - {user} ({size_kb:.1f} KB)")
                else:
                    print(f"  - {user}")
            
            return True
            
        except Exception as e:
            print(f"Failed to list users: {e}")
            return False
    
    def delete_user(self, name: str) -> bool:
        """
        Delete user enrollment.
        
        Args:
            name: User name to delete
            
        Returns:
            True if successful
        """
        try:
            if not validate_name(name):
                print(f"Error: Invalid name '{name}'")
                return False
            
            if not self.storage.enrollment_exists(name):
                print(f"Error: No enrollment found for user '{name}'")
                return False
            
            # Confirm deletion
            print(f"This will permanently delete the enrollment for '{name}'.")
            print("This action cannot be undone.")
            response = safe_input("Are you sure? (y/N): ")
            
            if response.lower() != 'y':
                print("Deletion cancelled.")
                return True
            
            # Delete enrollment
            success = self.storage.delete_enrollment(name)
            
            if success:
                print(f"Enrollment for '{name}' deleted successfully.")
                log_event(f"User '{name}' enrollment deleted")
            else:
                print(f"Failed to delete enrollment for '{name}'")
            
            return success
            
        except Exception as e:
            print(f"Failed to delete user: {e}")
            return False
    
    def test_camera(self, camera_index: int = 0) -> bool:
        """
        Test camera functionality.
        
        Args:
            camera_index: Camera device index
            
        Returns:
            True if camera test successful
        """
        try:
            import cv2
            
            print(f"Testing camera {camera_index}...")
            
            if not check_camera_available(camera_index):
                print(f"Camera {camera_index} not available.")
                return False
            
            cap = cv2.VideoCapture(camera_index)
            
            print("Camera test window opened. Press 'q' to quit.")
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Failed to read from camera.")
                    break
                
                # Add test overlay
                cv2.putText(frame, "Camera Test - Press 'q' to quit", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                cv2.imshow('Camera Test', frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
            
            cap.release()
            cv2.destroyAllWindows()
            
            print("Camera test completed.")
            return True
            
        except Exception as e:
            print(f"Camera test failed: {e}")
            return False


def create_parser() -> argparse.ArgumentParser:
    """Create command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="Face Unlock Prototype - Secure face-based authentication",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Enroll new user
  python face_unlock.py enroll --name alice --count 12
  
  # Verify user with liveness detection
  python face_unlock.py verify --name alice --require-liveness --callback "./unlock.sh"
  
  # List enrolled users
  python face_unlock.py list
  
  # Delete user enrollment
  python face_unlock.py delete --name alice
  
  # Test camera
  python face_unlock.py test-camera
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Enroll command
    enroll_parser = subparsers.add_parser('enroll', help='Enroll new user')
    enroll_parser.add_argument('--name', required=True, help='User name')
    enroll_parser.add_argument('--count', type=int, default=10, 
                              help='Number of frames to capture (default: 10)')
    enroll_parser.add_argument('--camera', type=int, default=0, 
                              help='Camera index (default: 0)')
    enroll_parser.add_argument('--out', default='face_data', 
                              help='Output directory (default: face_data)')
    
    # Verify command
    verify_parser = subparsers.add_parser('verify', help='Verify user identity')
    verify_parser.add_argument('--name', required=True, help='User name')
    verify_parser.add_argument('--threshold', type=float, default=0.55,
                              help='Distance threshold (default: 0.55)')
    verify_parser.add_argument('--camera', type=int, default=0,
                              help='Camera index (default: 0)')
    verify_parser.add_argument('--callback', help='Command to run on authorization')
    verify_parser.add_argument('--require-liveness', action='store_true',
                              help='Enable liveness detection')
    verify_parser.add_argument('--debug', action='store_true',
                              help='Enable debug mode')
    
    # List command
    subparsers.add_parser('list', help='List enrolled users')
    
    # Delete command
    delete_parser = subparsers.add_parser('delete', help='Delete user enrollment')
    delete_parser.add_argument('--name', required=True, help='User name')
    
    # Test camera command
    test_parser = subparsers.add_parser('test-camera', help='Test camera functionality')
    test_parser.add_argument('--camera', type=int, default=0,
                            help='Camera index (default: 0)')
    
    return parser


def main():
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    # Create CLI instance
    cli = FaceUnlockCLI()
    
    # Execute command
    success = False
    
    try:
        if args.command == 'enroll':
            success = cli.enroll_user(
                name=args.name,
                count=args.count,
                camera_index=args.camera,
                storage_dir=args.out
            )
        
        elif args.command == 'verify':
            success = cli.verify_user(
                name=args.name,
                threshold=args.threshold,
                camera_index=args.camera,
                callback_command=args.callback,
                require_liveness=args.require_liveness,
                debug=args.debug
            )
        
        elif args.command == 'list':
            success = cli.list_users()
        
        elif args.command == 'delete':
            success = cli.delete_user(args.name)
        
        elif args.command == 'test-camera':
            success = cli.test_camera(args.camera)
        
        else:
            print(f"Unknown command: {args.command}")
            parser.print_help()
            return 1
    
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
        return 1
    except Exception as e:
        print(f"Unexpected error: {e}")
        return 1
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
