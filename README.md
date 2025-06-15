# SenseToGive 🤖

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.7.0.72-green.svg)](https://opencv.org/)
[![Accessibility](https://img.shields.io/badge/Accessibility-Friendly-blueviolet.svg)](https://github.com/Fouad-Smaoui/SenseToGive)

## 🌟 Overview

SenseToGive is an inclusive, assistive project that uses a robotic arm (SO 100) to provide a unique interactive experience. When someone offers their hand, the robot responds by giving a candy, making it perfect for:

- 👩‍🦯 Blind or visually impaired users
- ♿ People with reduced mobility
- 🧠 Tech-assisted social interaction scenarios (e.g., festivals, hospitals, classrooms)

## 🎯 Impact

Our project aims to create meaningful interactions in various settings:

### Healthcare
- **Hospitals**: Provides a positive distraction for patients
- **Rehabilitation Centers**: Assists in physical therapy exercises
- **Care Homes**: Creates engaging social interactions

### Education
- **Classrooms**: Teaches robotics and accessibility
- **Special Education**: Supports inclusive learning environments
- **STEM Programs**: Demonstrates practical applications of robotics

### Public Spaces
- **Festivals**: Creates memorable interactive experiences
- **Museums**: Demonstrates assistive technology
- **Community Centers**: Promotes social inclusion

## 🛠️ Technical Details

### Prerequisites

- Python 3.10 or 3.11 (Python 3.13 is not supported)
- A webcam connected to your computer
- Robotic Arm SO 100

### Installation

1. **Clone the repository**:
   ```sh
   git clone https://github.com/Fouad-Smaoui/SenseToGive.git
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

## 🎮 Usage with Dataset

### 1. Dataset Structure
The project uses the `svla_so101_pickplace` dataset format. Your recordings will be saved in the following structure:
```
SenseToGive/
├── videos/
│   └── chunk-000/
│       ├── up/
│       │   └── episode_000001.mp4
│       └── side/
│           └── episode_000001.mp4
├── data/
│   └── chunk-000/
│       └── episode_000001.parquet
└── meta/
    └── info.json
```

### 2. Running the Hand Trigger
1. **Start the script**:
   ```sh
   python hand_trigger.py
   ```

2. **Controls**:
   - Press `r` to start/stop recording an episode
   - Press `q` to quit the application

3. **Recording Process**:
   - When recording starts, the script will:
     - Create video files in `videos/chunk-000/{up,side}/`
     - Save data files in `data/chunk-000/`
     - Update metadata in `meta/info.json`

4. **Data Format**:
   - **Videos**: MP4 format, 640x480 resolution, 30fps
   - **Parquet Files**: Include columns:
     - `frame_index`: Frame number in the episode
     - `episode_index`: Episode number
     - `timestamp`: Time since episode start
     - `action`: Robot state values
     - `observation.state`: Current robot state
     - `index`: Frame index
     - `task_index`: Task identifier

### 3. Using Recorded Data
1. **Viewing Episodes**:
   - Videos are saved in standard MP4 format
   - Use any video player to view recordings
   - Check `meta/info.json` for episode details

2. **Analyzing Data**:
   - Use pandas to read parquet files:
     ```python
     import pandas as pd
     df = pd.read_parquet('data/chunk-000/episode_000001.parquet')
     ```

3. **Dataset Integration**:
   - Recorded data follows the `svla_so101_pickplace` format
   - Compatible with existing dataset tools and analysis scripts
   - Can be used for training or testing machine learning models

## 🔧 Troubleshooting

- **Webcam Issues**: Ensure your webcam is connected and recognized by your system
- **Python Version**: Use Python 3.10 or 3.11. Python 3.13 is not supported
- **Dependencies**: If you encounter issues, try reinstalling the dependencies:
  ```sh
  pip install --upgrade pip setuptools wheel
  pip install -r requirements.txt
  ```

## 🤝 Contributing

We welcome contributions! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [svla_so101_pickplace](https://github.com/lerobot/svla_so101_pickplace) for the dataset structure and format
- The LeRobot Hackathon team for their support and guidance

## 📞 Contact

For questions or support, please open an issue in this repository.