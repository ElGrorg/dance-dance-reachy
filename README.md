# Dance Dance Reachy

This project uses a webcam and a YOLOv8-pose model to detect a person's pose in real-time and control a Reachy Mini robot to mirror their movements.

* **Head Sway:** The robot's head (lateral Y-axis) mirrors the user's hip sway relative to their shoulders.
* **Antennas:** The robot's antennas mirror the user's arm angles (relative to their torso).

## Features

* **Real-time Control:** Smooth, low-latency mirroring of human pose.
* **Producer-Consumer Architecture:** A multi-threaded design separates pose detection (producer) from robot control (consumer) for efficient processing.
* **Dynamic Calibration:** Press the 'c' key at any time to re-calibrate the "zero" position for your hip sway, allowing you to change your standing position.

## Installation

1.  **Clone the repository:**
    ```sh
    git clone [https://github.com/ElGrorg/dance-dance-reachy.git](https://github.com/ElGrorg/dance-dance-reachy.git)
    ```

2.  **Create and activate a virtual environment:**
    ```sh
    python -m venv .venv
    source .venv/bin/activate  # On Windows, use: venv\Scripts\activate
    ```

3.  **Install the dependencies:**
    ```sh
    pip install -r requirements.txt
    ```
    *Note: The `ultralytics` library will automatically download the `yolov8n-pose.pt` model on the first run.*

## Usage

1.  Ensure your Reachy Mini robot is connected and powered on.
2.  Ensure your webcam is connected.
3.  Run the daemon script:
    ```sh
    # This is for Mac, may be different on other devices
    mjpython -m reachy_mini.daemon.app.main --sim
4.  Run the main script:
    ```sh
    python main.py
    ```

5.  An OpenCV window will appear showing your webcam feed with pose annotations.

### Controls

* **`c` key:** Calibrates the hip sway. Stand in a neutral, straight-on position and press 'c' to set this as the zero point.
* **`q` key:** Quits the application and safely stops all threads.

## Project Structure

* `main.py`: The main entry point of the application. It initializes the robot, queues, and threads, and runs the main UI loop (OpenCV window).
* `src/pose_detector.py`: Contains the `yolo_loop` (producer thread) responsible for all webcam capture and YOLO pose estimation.
* `src/robot_controller.py`: Contains the `control_reachy` (consumer thread) responsible for mapping pose data to robot commands.
* `src/utils.py`: Contains helper functions, such as `calculate_angle`.
* `tests/`: Contains unit tests for the project.
    * `test_utils.py`: Tests the `calculate_angle` function.
