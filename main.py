import cv2
import threading
import queue
from reachy_mini import ReachyMini
from reachy_mini.utils import create_head_pose

from src.pose_detector import yolo_loop
from src.robot_controller import control_reachy

def main():
    """
    Initializes and runs the Reachy Pose Controller application.
    
    Sets up the robot, communication queues, and worker threads (pose detection
    and robot control). Also manages the main UI loop (OpenCV window) for
    displaying video and handling user input.
    """
    
    # --- Robot Initialization ---
    try:
        mini = ReachyMini()
    except Exception as e:
        print(f"Error initializing Reachy Mini: {e}")
        print("Please check the robot's connection and try again.")
        return

    # Set a default starting position
    mini.goto_target(head=create_head_pose(y=0, mm=True))

    # --- Communication Queues ---
    # frame_queue: Passes video frames from detector to main thread for display
    frame_queue = queue.Queue(maxsize=2)
    # pose_queue: Passes processed pose data from main thread to robot controller
    pose_queue = queue.Queue(maxsize=10)

    # --- Threading Events ---
    stop_signal = threading.Event()
    calibrate_event = threading.Event()

    # --- Calibration State ---
    # This dict holds the "zero" values for all calibrated poses
    pose_zero_offsets = {'hip_sway': 0.0}

    # --- Thread Creation ---
    print("Starting threads...")

    yolo_thread = threading.Thread(
        target=yolo_loop,
        args=(frame_queue, stop_signal, calibrate_event, pose_zero_offsets),
        daemon=True,
        name="YOLO_Thread"
    )

    consumer_thread = threading.Thread(
        target=control_reachy,
        args=(mini, pose_queue, stop_signal),
        daemon=True,
        name="Reachy_Thread"
    )

    yolo_thread.start()
    consumer_thread.start()

    # --- MAIN THREAD LOOP (for UI) ---
    print("Starting UI loop in main thread.")
    print("--- Press 'c' to calibrate hip sway to zero ---")
    print("--- Press 'q' to quit ---")

    try:
        while not stop_signal.is_set():
            try:
                # Get annotated frame and pose data from the YOLO thread
                annotated_frame, latest_pose_data = frame_queue.get(timeout=0.1)

                # Pass pose data to the robot controller thread
                try:
                    pose_queue.put_nowait(latest_pose_data)
                except queue.Full:
                    # If controller is busy, just drop the data
                    pass

                cv2.imshow("Reachy Mini Pose Controller", annotated_frame)

                # --- User Input ---
                key = cv2.waitKey(1) & 0xFF

                if key == ord('q'):
                    print("'q' pressed, stopping all threads...")
                    stop_signal.set()
                    break

                elif key == ord('c'):
                    print("'c' pressed, sending calibration signal...")
                    calibrate_event.set()

            except queue.Empty:
                # No new frame, just keep looping
                continue

    except KeyboardInterrupt:
        print("KeyboardInterrupt, stopping all threads...")
        stop_signal.set()

    # --- Cleanup ---
    print("Waiting for threads to join...")
    yolo_thread.join()
    consumer_thread.join()
    cv2.destroyAllWindows()
    print("Program finished.")


if __name__ == "__main__":
    main()