# SenseToGive

SenseToGive is an inclusive, assistive project that uses a robotic arm (SO 100) to provide a unique interactive experience. When someone offers their hand, the robot responds by giving a candy, making it perfect for:

- 👩‍🦯 Blind or visually impaired users
- ♿ People with reduced mobility
- 🧠 Tech-assisted social interaction scenarios (e.g., festivals, hospitals, classrooms)

## Prerequisites

- Python 3.10 or 3.11 (Python 3.13 is not supported due to compatibility issues with NumPy, OpenCV, and pandas).
- A webcam connected to your computer.

## Installation

1. **Clone the repository** (if you haven't already):
   ```sh
   git clone https://github.com/yourusername/SenseToGive.git
   cd SenseToGive
   ```

2. **Create a virtual environment**:
   ```sh
   python -m venv venv
   ```

3. **Activate the virtual environment**:
   - On Windows:
     ```sh
     .\venv\Scripts\activate
     ```
   - On macOS/Linux:
     ```sh
     source venv/bin/activate
     ```

4. **Install dependencies**:
   ```sh
   pip install -r requirements.txt
   ```

## Usage

1. **Run the hand trigger script**:
   ```sh
   python hand_trigger.py
   ```

2. **Controls**:
   - Press `r` to start/stop recording an episode.
   - Press `q` to quit the application.

3. **Recording**:
   - When recording is active, the script saves:
     - Video files in `videos/chunk-000/{up,side}/episode_{:06d}.mp4`.
     - Data files in `data/chunk-000/episode_{:06d}.parquet`.
     - Metadata in `meta/info.json`.

4. **Hand Detection**:
   - The script uses OpenCV to detect hand gestures based on skin color and motion.
   - When a hand is detected, the robot state is updated and, if triggered, the episode is recorded.

## Dataset Compatibility

The recorded episodes are fully compatible with the `svla_so101_pickplace` dataset:
- **Directory Structure**: Matches the dataset's expected paths.
- **Data Format**: Parquet files include all required columns (`frame_index`, `episode_index`, `timestamp`, `action`, `observation.state`, `index`, `task_index`).
- **Metadata**: Written to `meta/info.json` with details about episodes, frames, and features.

## Impact

SenseToGive is designed to create an inclusive, assistive experience that aligns perfectly with real-world impact, especially for:

- 👩‍🦯 Blind or visually impaired users
- ♿ People with reduced mobility
- 🧠 Tech-assisted social interaction scenarios (e.g., festivals, hospitals, classrooms)

## Troubleshooting

- **Webcam Issues**: Ensure your webcam is connected and recognized by your system.
- **Python Version**: Use Python 3.10 or 3.11. Python 3.13 is not supported.
- **Dependencies**: If you encounter issues, try reinstalling the dependencies:
  ```sh
  pip install --upgrade pip setuptools wheel
  pip install -r requirements.txt
  ```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [svla_so101_pickplace](https://github.com/lerobot/svla_so101_pickplace) for the dataset structure and format. 