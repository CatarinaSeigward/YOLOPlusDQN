# Deep Reinforcement Learning for Automated Boss Battle Gaming

A hybrid YOLO-DQN approach for automated boss battle gameplay in action games, demonstrated on Black Myth: Wukong's Wandering Wight boss.

## Overview

This project combines real-time action recognition using YOLOv8 with Deep Q-Network (DQN) reinforcement learning to create an AI agent capable of learning and defeating bosses in action games. The system processes visual game data to recognize boss attack patterns and makes strategic combat decisions in real-time.

### Key Features

- **Real-time Boss Action Recognition**: YOLOv8 model detecting 9 distinct boss actions
- **DQN-based Decision Making**: Reinforcement learning agent with streamlined action space
- **Visual State Processing**: Dual-stream input processing for action detection and health monitoring
- **Efficient Learning**: Successfully defeats boss within 60 training episodes
- **Modular Architecture**: Easily extensible for different games and bosses

## Architecture

```
Game Screen Capture
        │
        ├──► Boss Action Detection (YOLOv8)
        │           │
        │           └──► State Construction ──► DQN Agent ──► Action
        │                                            │
        └──► Health Bar Monitoring                   │
                    │                                 │
                    └──► Reward Calculation ◄────────┘
```

##  Requirements

### System Requirements
- **GPU**: NVIDIA GPU with CUDA support (highly recommended for real-time inference)
  - Minimum 4GB VRAM for YOLOv8 inference
  - 6GB+ VRAM recommended for training
- **Game**: Black Myth: Wukong (or any target game with lock-on combat system)
- **RAM**: 8GB minimum, 16GB+ recommended

### Technology Stack

#### Core Frameworks
- **PyTorch** (>=2.0.0): Deep learning framework for DQN implementation
- **Ultralytics YOLOv8** (>=8.0.0): Real-time object detection for boss action recognition
- **OpenCV** (>=4.5.0): Computer vision library for image processing and screen capture

#### Windows Integration
- **PyWin32** (>=300): Windows API access for screen capture (win32gui, win32ui, win32con)
- **Keyboard** (>=0.13.5): Keyboard input simulation and event monitoring

#### Data Processing & Visualization
- **NumPy** (>=1.21.0): Numerical computing and array operations
- **Matplotlib** (>=3.3.0): Training visualization and loss curve plotting

#### Optional Development Tools
- **Jupyter** (>=1.0.0): Interactive development and experimentation
- **IPython** (>=8.0.0): Enhanced Python shell for debugging

## 🚀 Installation

### Step 1: Clone the Repository
### Step 2: Set Up Python Environment

#### Option A: Using Virtual Environment (Recommended)
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On Linux/Mac:
source venv/bin/activate
```

#### Option B: Using Conda
```bash
# Create conda environment
conda create -n yolodqn python=3.10
conda activate yolodqn
```

### Step 3: Install CUDA and PyTorch

**For GPU Training (Recommended):**
```bash
# Install PyTorch with CUDA 11.8 (check your CUDA version first)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

**For CPU Only (Not Recommended for Training):**
```bash
pip install torch torchvision
```

**Verify CUDA Installation:**
```python
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'CUDA Version: {torch.version.cuda}')"
```

### Step 4: Install Project Dependencies
```bash
pip install -r requirements.txt
```

This will install:
- ultralytics (YOLOv8)
- opencv-python
- numpy
- pywin32
- keyboard
- matplotlib
- jupyter (optional)
- ipython (optional)

### Step 5: Download Pre-trained YOLO Model

Download the pre-trained boss detection model and place it in the project root directory:

```bash
# Option 1: Download from releases
# Place boss_only.pt in the project root

# Option 2: Train your own model (see Training YOLO section)
```

### Step 6: Verify Installation

```bash
# Test CUDA availability
python -c "import torch; print('CUDA:', torch.cuda.is_available())"

# Test YOLO import
python -c "from ultralytics import YOLO; print('YOLO imported successfully')"

# Test screen capture
python -c "from grabscreen import grab_screen; print('Screen capture working')"
```

### Troubleshooting Installation

**Issue: PyWin32 Import Error**
```bash
# Run post-install script
python venv/Scripts/pywin32_postinstall.py -install
```

**Issue: CUDA Out of Memory**
- Reduce batch size in training configuration
- Use a smaller YOLO model variant (yolov8n.pt instead of yolov8m.pt)

**Issue: Keyboard Module Not Working**
- Run Python as Administrator
- Check antivirus software blocking keyboard simulation

**Issue: OpenCV Import Error**
```bash
pip uninstall opencv-python opencv-python-headless
pip install opencv-python
```

## Project Structure

```
dqn-boss-battle/
│
├── train.py                 # Main training script
├── test.py                  # Testing/evaluation script
├── dqn_agent.py            # DQN agent implementation
├── game_environment.py      # Game environment wrapper
├── directkeys.py           # Keyboard input simulation
├── grabscreen.py           # Screen capture utilities
├── locationfinder.py       # Blood bar location helper
├── Loss_Visualization.py   # Training visualization
│
├── boss_only.pt            # Pre-trained YOLO model (not included)
├── checkpoints/            # Saved model checkpoints
│   ├── best_model.pth
│   └── checkpoint_*.pth
│
└── README.md
```

##  Usage

### Training the Agent

1. **Start the game and navigate to the boss fight**

2. **Run the training script**
```bash
python train.py
```

3. **Control the training process**
   - Press `T` to start the training program
   - Press `O` to start a new episode
   - Press `R` to end the current episode
   - Press `Q` to quit training

The agent will automatically learn through trial and error, saving checkpoints every 10 episodes.

### Testing the Trained Model

```bash
python test.py
```
- Press `T` to start testing
- Press `Q` to quit

### Monitoring Training Progress

```bash
python Loss_Visualization.py
```

This will generate a loss curve visualization showing the agent's learning progress.

## 🔧 Configuration

### DQN Parameters (in `dqn_agent.py`)

```python
self.gamma = 0.95           # Discount factor
self.epsilon = 1.0          # Initial exploration rate
self.epsilon_min = 0.01     # Minimum exploration rate
self.epsilon_decay = 0.995  # Exploration decay rate
self.learning_rate = 0.001  # Learning rate
self.memory = deque(maxlen=2000)  # Experience replay buffer
```

### Action Space (in `game_environment.py`)

```python
self.actions = {
    0: attack,     # Light attack
    1: dodge2,     # Forward dodge
    2: attack3     # Heavy attack
}
```

### Screen Configuration

```python
self.screen_region = (1, 45, 1282, 767)  # Game window region
self.resize_width = 640                  # Resized width for processing
self.resize_height = 384                 # Resized height for processing
```

## 📊 Performance

- **Training Episodes to Victory**: ~60 episodes
- **Average Episode Length**: 200-300 steps
- **Success Rate**: Consistently defeats boss after training
- **Processing Speed**: ~10 FPS (limited by game interaction)

## Boss Action Recognition

The YOLOv8 model is trained to detect the following boss actions:

1. Jump Attack
2. Feet Attack
3. Clap Attack
4. Far Attack
5. Boom Attack
6. Sweep Attack
7. Crying Animation
8. Downed State
9. Standing Position

## Technical Details

### State Representation
- **Action Class**: Detected boss action (0-8, 99 for no action)
- **Confidence Score**: YOLO detection confidence (0.0-1.0)
- **Boss Health**: Normalized boss HP value
- **Player Health**: Normalized player HP value

### Reward Function
```python
reward = 0
if boss_blood_change > 0:
    reward += boss_blood_change * 30  # Reward for damaging boss
if player_blood_change > 0:
    reward -= player_blood_change * 5  # Penalty for taking damage
```

## Known Limitations

- Requires pre-trained YOLO model for each boss
- Windows-only due to DirectX screen capture
- Limited to games with lock-on targeting systems
- Training data scalability challenges

## Future Work

- Integration with Grounding DINO for zero-shot boss detection
- Multi-boss generalization capabilities
- Cross-platform support
- Real-time strategy adaptation

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

##  Acknowledgments

- **analoganddigital** - For the DQN_play_sekiro project inspiration
- **FAN XU** - For the pre-trained YOLOv8 model for Black Myth: Wukong
- **Ultralytics** - For the YOLOv8 framework

##  License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Disclaimer

This project is for educational and research purposes only. Please respect the game developers' terms of service and use responsibly.

---

**Author**: Kaiwen Lin  
**Email**: kaiwenlin@utexas.edu  
**Institution**: University of Texas at Austin
