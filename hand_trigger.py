import cv2
import numpy as np
import time
import os
import json
import pandas as pd
from datetime import datetime
from typing import Tuple, Optional, Dict, List

class HandDetector:
    """
    A class to handle real-time hand detection using OpenCV.
    This class manages the hand detection pipeline and records data in the svla_so101_pickplace format.
    """
    def __init__(self, 
                 debounce_time: float = 0.5):
        """
        Initialize the hand detector with OpenCV.
        
        Args:
            debounce_time: Time in seconds to wait before triggering (prevents false triggers)
        """
        # Initialize debounce-related variables
        self.debounce_time = debounce_time
        self.last_trigger_time = 0
        self.hand_present = False
        self.hand_present_start_time = None

        # Initialize recording-related variables
        self.current_episode = 0
        self.is_recording = False
        self.video_writers = {}  # Dictionary to store video writers for different views
        self.episode_start_time = None
        self.episode_data = []
        
        # Create recording directory structure
        self._setup_recording_dirs()
        
        # Initialize background subtractor for motion detection
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=20, varThreshold=25)
        
        # Initialize robot state (6-DOF arm)
        self.robot_state = {
            'shoulder_pan.pos': 0.0,
            'shoulder_lift.pos': 0.0,
            'elbow_flex.pos': 0.0,
            'wrist_flex.pos': 0.0,
            'wrist_roll.pos': 0.0,
            'gripper.pos': 0.0
        }

    def _setup_recording_dirs(self):
        """Set up the recording directory structure to match the dataset format."""
        # Create main directories
        os.makedirs(os.path.join("videos", "chunk-000", "up"), exist_ok=True)
        os.makedirs(os.path.join("videos", "chunk-000", "side"), exist_ok=True)
        os.makedirs(os.path.join("data", "chunk-000"), exist_ok=True)
        os.makedirs(os.path.join("meta"), exist_ok=True)

    def detect_hand(self, frame: cv2.Mat) -> Tuple[bool, Optional[Tuple[int, int, int, int]]]:
        """
        Detect if a hand is present in the frame using color and motion analysis.
        
        Args:
            frame: Input frame from webcam in BGR format
            
        Returns:
            Tuple containing:
            - Boolean indicating if a hand is present
            - Optional tuple of (x_min, y_min, x_max, y_max) for the bounding box
        """
        # Convert to HSV color space for better skin color detection
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Define range for skin color in HSV
        lower_skin = np.array([0, 20, 70], dtype=np.uint8)
        upper_skin = np.array([20, 255, 255], dtype=np.uint8)
        
        # Create a mask for skin color
        skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)
        
        # Apply morphological operations to clean up the mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        skin_mask = cv2.erode(skin_mask, kernel, iterations=2)
        skin_mask = cv2.dilate(skin_mask, kernel, iterations=2)
        
        # Apply Gaussian blur to reduce noise
        skin_mask = cv2.GaussianBlur(skin_mask, (3, 3), 0)
        
        # Find contours in the mask
        contours, _ = cv2.findContours(skin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Find the largest contour (assumed to be the hand)
            largest_contour = max(contours, key=cv2.contourArea)
            
            # Calculate area of the contour
            area = cv2.contourArea(largest_contour)
            
            # Only consider it a hand if the area is large enough
            if area > 1000:  # Adjust this threshold based on your setup
                # Get bounding box
                x, y, w, h = cv2.boundingRect(largest_contour)
                
                # Draw the contour and bounding box
                cv2.drawContours(frame, [largest_contour], -1, (0, 255, 0), 2)
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                
                return True, (x, y, x+w, y+h)
        
        return False, None

    def control_robot(self, frame: cv2.Mat, bbox: Tuple[int, int, int, int]):
        """
        Update robot state based on hand position.
        
        Args:
            frame: Current camera frame
            bbox: Bounding box of detected hand (x_min, y_min, x_max, y_max)
        """
        # Calculate hand center position
        x_min, y_min, x_max, y_max = bbox
        hand_center = ((x_min + x_max) // 2, (y_min + y_max) // 2)
        
        # Calculate relative position (0-1 range)
        h, w = frame.shape[:2]
        rel_x = hand_center[0] / w
        rel_y = hand_center[1] / h
        
        # Update robot state based on hand position
        # These are simplified mappings - adjust based on your robot's kinematics
        self.robot_state['shoulder_pan.pos'] = rel_x * 2 - 1  # Map to [-1, 1]
        self.robot_state['shoulder_lift.pos'] = rel_y * 2 - 1
        self.robot_state['elbow_flex.pos'] = (rel_x + rel_y) / 2
        self.robot_state['wrist_flex.pos'] = rel_y
        self.robot_state['wrist_roll.pos'] = rel_x
        self.robot_state['gripper.pos'] = 1.0 if self.hand_present else 0.0

    def start_recording(self):
        """Start recording a new episode."""
        if not self.is_recording:
            self.current_episode += 1
            self.is_recording = True
            self.episode_start_time = time.time()
            self.episode_data = []
            
            # Initialize video writers for both views
            for view in ['up', 'side']:
                video_path = os.path.join(
                    "videos", "chunk-000", view, f"episode_{self.current_episode:06d}.mp4"
                )
                
                # Create video writer with H.264 codec
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use mp4v codec
                self.video_writers[view] = cv2.VideoWriter(
                    video_path,
                    fourcc,
                    30,  # 30 fps
                    (640, 480)  # 640x480 resolution
                )
            
            print(f"Starting episode {self.current_episode}")

    def stop_recording(self):
        """Stop recording the current episode and save data."""
        if self.is_recording:
            self.is_recording = False
            
            # Close video writers
            for writer in self.video_writers.values():
                writer.release()
            self.video_writers = {}
            
            # Save episode data as parquet file
            df = pd.DataFrame(self.episode_data)
            data_path = os.path.join(
                "data", "chunk-000", f"episode_{self.current_episode:06d}.parquet"
            )
            df.to_parquet(data_path)
            
            # Update metadata
            self._update_metadata()
            
            print(f"Episode {self.current_episode} saved")

    def _update_metadata(self):
        """Update the metadata file with episode information."""
        meta_path = os.path.join("meta", "info.json")
        
        # Create or load existing metadata
        if os.path.exists(meta_path):
            with open(meta_path, 'r') as f:
                metadata = json.load(f)
        else:
            metadata = {
                "codebase_version": "v1.0",
                "robot_type": "so100_follower",
                "total_episodes": 0,
                "total_frames": 0,
                "total_tasks": 1,
                "total_videos": 0,
                "fps": 30,
                "features": {
                    "action": {
                        "dtype": "float32",
                        "shape": [6],
                        "names": list(self.robot_state.keys())
                    },
                    "observation.state": {
                        "dtype": "float32",
                        "shape": [6],
                        "names": list(self.robot_state.keys())
                    },
                    "observation.images.up": {
                        "dtype": "video",
                        "shape": [480, 640, 3],
                        "info": {
                            "video.height": 480,
                            "video.width": 640,
                            "video.codec": "mp4v",
                            "video.pix_fmt": "yuv420p",
                            "video.is_depth_map": False,
                            "video.fps": 30,
                            "video.channels": 3,
                            "has_audio": False
                        }
                    },
                    "observation.images.side": {
                        "dtype": "video",
                        "shape": [480, 640, 3],
                        "info": {
                            "video.height": 480,
                            "video.width": 640,
                            "video.codec": "mp4v",
                            "video.pix_fmt": "yuv420p",
                            "video.is_depth_map": False,
                            "video.fps": 30,
                            "video.channels": 3,
                            "has_audio": False
                        }
                    }
                }
            }
        
        # Update metadata
        metadata["total_episodes"] = self.current_episode
        metadata["total_frames"] += len(self.episode_data)
        metadata["total_videos"] = self.current_episode * 2  # Two views per episode
        
        # Save updated metadata
        with open(meta_path, 'w') as f:
            json.dump(metadata, f, indent=4)

    def should_trigger(self, hand_present: bool) -> bool:
        """
        Determine if the robot should be triggered based on debounce logic.
        
        Args:
            hand_present: Whether a hand is currently detected
            
        Returns:
            True if the robot should be triggered, False otherwise
        """
        current_time = time.time()
        
        if hand_present:
            if not self.hand_present:
                self.hand_present = True
                self.hand_present_start_time = current_time
            elif (current_time - self.hand_present_start_time) >= self.debounce_time:
                if (current_time - self.last_trigger_time) >= self.debounce_time:
                    self.last_trigger_time = current_time
                    return True
        else:
            self.hand_present = False
            self.hand_present_start_time = None
            
        return False

def main():
    """
    Main function to run the hand detection system.
    Handles webcam initialization, frame processing, and visualization.
    """
    # Initialize webcam (0 is usually the default webcam)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return

    # Set webcam resolution to match dataset
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

    # Initialize hand detector with default parameters
    detector = HandDetector()
    
    print("Hand detection started. Press:")
    print("'r' to start/stop recording")
    print("'q' to quit")
    
    frame_count = 0
    while True:
        # Read frame from webcam
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame")
            break
            
        # Detect hand in the current frame
        hand_present, bbox = detector.detect_hand(frame)
        
        # Add visual feedback if hand is detected
        if hand_present and bbox:
            x_min, y_min, x_max, y_max = bbox
            # Green box if ready to trigger, yellow if still in debounce period
            color = (0, 255, 0) if detector.should_trigger(hand_present) else (0, 255, 255)
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 2)
            
            if detector.should_trigger(hand_present):
                print("Triggering robot arm...")
                detector.control_robot(frame, bbox)
        
        # Add recording status to frame
        if detector.is_recording:
            cv2.putText(frame, f"Recording Episode {detector.current_episode}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            # Record frame data
            frame_data = {
                'frame_index': frame_count,
                'episode_index': detector.current_episode,
                'timestamp': time.time() - detector.episode_start_time,
                'action': list(detector.robot_state.values()),
                'observation.state': list(detector.robot_state.values()),
                'index': frame_count,  # index is usually the same as frame_index
                'task_index': 0       # single task for this demo
            }
            detector.episode_data.append(frame_data)
            
            # Write frame to both video views (using same frame for both views in this demo)
            for writer in detector.video_writers.values():
                writer.write(frame)
            
            frame_count += 1
        
        # Display the processed frame
        cv2.imshow('Hand Detection', frame)
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            if detector.is_recording:
                detector.stop_recording()
            break
        elif key == ord('r'):
            if detector.is_recording:
                detector.stop_recording()
            else:
                detector.start_recording()
                frame_count = 0
    
    # Clean up resources
    if detector.video_writers:
        for writer in detector.video_writers.values():
            writer.release()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 