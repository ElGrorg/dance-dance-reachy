import cv2
from ultralytics import YOLO # type: ignore
import queue
import threading
from typing import Dict, Optional

from .utils import calculate_angle

# --- COCO Keypoint Indices ---
LEFT_SHOULDER = 5
RIGHT_SHOULDER = 6
LEFT_ELBOW = 7
RIGHT_ELBOW = 8
LEFT_HIP = 11
RIGHT_HIP = 12
CONF_THRESHOLD = 0.5

def yolo_loop(
    frame_queue: queue.Queue, 
    stop_event: threading.Event, 
    calibrate_event: threading.Event, 
    pose_zero_offsets: dict
):
    """
    Producer thread function.
    
    Initializes a YOLOv8-pose model and runs a loop to capture video,
    perform pose estimation, calculate angles/sway, and put the
    annotated frame and pose data into the shared queue.
    
    Args:
        frame_queue: Queue to put (annotated_frame, pose_data) tuples.
        stop_event: Event to signal when the thread should stop.
        calibrate_event: Event to signal when to recalibrate pose.
        pose_zero_offsets: Dictionary to store and update the 'zero'
                           offset for hip sway.
    """
    
    model = YOLO('yolov8n-pose.pt')
    
    try:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open video source.")
            stop_event.set()
            return
    except Exception as e:
        print(f"Error opening camera: {e}")
        stop_event.set()
        return

    print("Starting webcam feed processing...")

    while cap.isOpened() and not stop_event.is_set():
        success, frame = cap.read()
        if not success:
            print("Error: Failed to read frame.")
            break

        # Run YOLO model
        results = model(frame, verbose=False)
        annotated_frame = results[0].plot()

        latest_pose_data: Dict[str, Optional[float]] = {
            "left_arm": None,
            "right_arm": None,
            "hip_sway": None,
        }

        try:
            # Get keypoints for the first detected person
            all_keypoints_data = results[0].keypoints.data.cpu().numpy()
            if len(all_keypoints_data) == 0:
                continue # No person detected

            person_kpts = all_keypoints_data[0] # Process only one person

            # --- Get all required keypoints ---
            l_hip_kpt = person_kpts[LEFT_HIP]
            r_hip_kpt = person_kpts[RIGHT_HIP]
            l_shoulder_kpt = person_kpts[LEFT_SHOULDER]
            r_shoulder_kpt = person_kpts[RIGHT_SHOULDER]
            l_elbow_kpt = person_kpts[LEFT_ELBOW]
            r_elbow_kpt = person_kpts[RIGHT_ELBOW]

            # --- Left Arm Angle ---
            if (l_hip_kpt[2] > CONF_THRESHOLD and
                l_shoulder_kpt[2] > CONF_THRESHOLD and
                l_elbow_kpt[2] > CONF_THRESHOLD):
                
                l_hip_pos = l_hip_kpt[:2]
                l_shoulder_pos = l_shoulder_kpt[:2]
                l_elbow_pos = l_elbow_kpt[:2]
                
                latest_pose_data["left_arm"] = calculate_angle(l_hip_pos, l_shoulder_pos, l_elbow_pos)
                
                cv2.putText(annotated_frame, f"L: {latest_pose_data['left_arm']:.1f}",
                            (int(l_shoulder_pos[0]), int(l_shoulder_pos[1] - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # --- Right Arm Angle ---
            if (r_hip_kpt[2] > CONF_THRESHOLD and
                r_shoulder_kpt[2] > CONF_THRESHOLD and
                r_elbow_kpt[2] > CONF_THRESHOLD):
                
                r_hip_pos = r_hip_kpt[:2]
                r_shoulder_pos = r_shoulder_kpt[:2]
                r_elbow_pos = r_elbow_kpt[:2]
                
                latest_pose_data["right_arm"] = calculate_angle(r_hip_pos, r_shoulder_pos, r_elbow_pos)
                
                cv2.putText(annotated_frame, f"R: {latest_pose_data['right_arm']:.1f}",
                            (int(r_shoulder_pos[0]), int(r_shoulder_pos[1] - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # --- Hip Sway Calculation ---
            if (l_hip_kpt[2] > CONF_THRESHOLD and
                r_hip_kpt[2] > CONF_THRESHOLD and
                l_shoulder_kpt[2] > CONF_THRESHOLD and
                r_shoulder_kpt[2] > CONF_THRESHOLD):

                l_shoulder_x, r_shoulder_x = l_shoulder_kpt[0], r_shoulder_kpt[0]
                l_hip_x, r_hip_x = l_hip_kpt[0], r_hip_kpt[0]

                shoulder_center_x = (l_shoulder_x + r_shoulder_x) / 2
                hip_center_x = (l_hip_x + r_hip_x) / 2

                raw_hip_sway = hip_center_x - shoulder_center_x
                

                # Check for calibration signal
                if calibrate_event.is_set():
                    pose_zero_offsets['hip_sway'] = raw_hip_sway
                    calibrate_event.clear()
                    print(f"--- HIP SWAY CALIBRATED: Zero set to {raw_hip_sway:.1f} pixels ---")                
                
                # Calculate and store the final, relative sway
                final_hip_sway = raw_hip_sway - pose_zero_offsets['hip_sway']
                latest_pose_data["hip_sway"] = final_hip_sway

                cv2.putText(annotated_frame, f"Sway: {final_hip_sway:.1f}",
                            (int(hip_center_x), int(l_hip_kpt[1] - 10)), # Draw near left hip
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        except Exception as e:  # noqa: F841
            # print(f"Error during keypoint processing: {e}") # Uncomment for debugging
            pass

        # --- Put data in the queue for the main thread ---
        try:
            frame_queue.put_nowait((annotated_frame, latest_pose_data.copy()))
        except queue.Full:
            # If main thread is slow, drop the frame
            pass

    # Release resources
    cap.release()
    print("YOLO loop stopping.")