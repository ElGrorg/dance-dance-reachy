import numpy as np
import queue
import math
import threading
from reachy_mini import ReachyMini
from reachy_mini.utils import create_head_pose

# --- Mapping Parameters ---
# Map hip sway (in pixels) to the head's lateral movement (in mm)
# Adjust SWAY_PIXEL_MAX based on your camera and distance
SWAY_PIXEL_MAX = 80  # Max pixel offset left/right
HEAD_Y_MM_MAX = 35   # Max head movement left/right in mm

# Map pixel range to head mm range
# Note: We invert the head range to "mirror" you.
# Your move right (positive pixels) = Reachy's head right (negative mm)
SWAY_INPUT_RANGE = [-SWAY_PIXEL_MAX, SWAY_PIXEL_MAX]
HEAD_OUTPUT_RANGE = [HEAD_Y_MM_MAX, -HEAD_Y_MM_MAX] # Mirrored

def control_reachy(
    mini: ReachyMini, 
    pose_queue: queue.Queue, 
    stop_event: threading.Event
):
    """
    Consumer thread function.
    
    Gets pose data from the queue, maps the values to robot commands
    (head sway, antenna angles), and sends them to the Reachy Mini.
    
    Args:
        mini: The initialized ReachyMini object.
        pose_queue: Queue to get (pose_data) dictionaries from.
        stop_event: Event to signal when the thread should stop.
    """
    print("Starting Reachy control loop. Waiting for pose data...")

    # Store the last valid commands to resend if new data isn't available
    last_head_pose = create_head_pose(y=0, mm=True)
    last_antennas = [0.0, 0.0]  # Neutral antenna position (straight up)

    while not stop_event.is_set():
        try:
            # Get the latest full pose data packet
            pose_data = pose_queue.get(timeout=0.1)
            
            new_head_pose = last_head_pose
            new_antennas = last_antennas

            # --- 1. Update Head Command ---
            if pose_data['hip_sway'] is not None:
                # Map the relative pixel sway to head Y-position in mm
                head_y_cmd = np.interp(
                    pose_data['hip_sway'],
                    SWAY_INPUT_RANGE,
                    HEAD_OUTPUT_RANGE
                )
                # Clip the value to ensure it's within safe limits
                head_y_cmd = np.clip(head_y_cmd, -HEAD_Y_MM_MAX, HEAD_Y_MM_MAX)
                
                # Create the head pose object
                new_head_pose = create_head_pose(y=head_y_cmd, mm=True)
                
            # --- 2. Update Antenna Commands ---
            if pose_data['left_arm'] is not None and pose_data['right_arm'] is not None:
                # Invert angles (math.pi - angle) so "arm up" = "antenna up"
                l_angle_cmd = math.pi - pose_data['left_arm']
                r_angle_cmd = math.pi - pose_data['right_arm']
                
                # Create the antenna command list
                new_antennas = [float(-l_angle_cmd), float(r_angle_cmd)]
            
            # --- 3. Send Commands to Reachy ---
            mini.set_target(
                head=new_head_pose,
                antennas=new_antennas
            )

            # Store these as the last valid commands
            last_head_pose = new_head_pose
            last_antennas = new_antennas
            
            pose_queue.task_done()

        except queue.Empty:
            # No new data.
            # We could optionally resend the last command to maintain pose,
            # but for now, we just continue.
            continue
        
        except Exception as e:
            print(f"Error in control loop: {e}")

    print("Reachy control loop stopping.")
    # Set a final neutral position on exit
    mini.goto_target(
        head=create_head_pose(y=0, mm=True),
        antennas=[0.0, 0.0]
    )