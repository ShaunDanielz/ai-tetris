# AI Tetris

## Project Overview
AI Tetris is an implementation of the classic Tetris game enhanced with artificial intelligence. The project leverages machine learning, specifically TensorFlow, to train an AI agent to play Tetris autonomously. The AI is trained using a combination of game logic (implemented with Pygame) and a neural network model (built with TensorFlow) to optimize gameplay strategies.

This project aims to:
- Demonstrate reinforcement learning or supervised learning techniques applied to a game environment.
- Utilize GPU acceleration for faster training where available.
- Provide a fun and educational example of AI in action.

---

## Prerequisites
Before setting up the project, ensure you have the following installed:
- **Anaconda** or **Miniconda**: For managing Python environments and dependencies.
- **NVIDIA GPU** (optional, for GPU acceleration): Requires compatible drivers installed.
- **Windows**: This setup is tested on Windows, but it can be adapted for other platforms.

---

## Environment Setup with Conda
To ensure a consistent and isolated environment, this project uses Conda to manage dependencies. Follow these steps to set up the environment with Python 3.10 and TensorFlow 2.10.0, optimized for GPU support.

### Step 1: Clone the Repository
Clone this repository to your local machine:
```bash
git clone https://github.com/ShaunDanielz/ai-tetris.git
cd ai-tetris
```

### Step 2: Create the Conda Environment
Create a new Conda environment named `tetris_ai` with Python 3.10:
```bash
conda create --name tetris_ai python=3.10
```

### Step 3: Activate the Environment
Activate the environment to begin working in it:
```bash
conda activate tetris_ai
```

### Step 4: Install Dependencies
Install the required packages, including TensorFlow with GPU support, Pygame, and other dependencies:

#### Core Dependencies
```bash
conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1 tensorflow-gpu=2.10.0 pygame
pip install "numpy<2.0" psutil
```

- **TensorFlow 2.10.0**: Chosen for its stability and GPU support with CUDA 11.2.
- **CUDA 11.2 and cuDNN 8.1**: Required for TensorFlow 2.10.0 GPU support, installed via Conda to avoid system-wide conflicts.
- **NumPy <2.0**: TensorFlow 2.10.0 is incompatible with NumPy 2.x due to breaking changes; this ensures compatibility.
- **Pygame**: Used for the Tetris game environment.
- **psutil**: Optional utility for monitoring system resources (e.g., CPU usage).

#### Why These Versions?
- TensorFlow 2.10.0 is the last version to support GPU acceleration on native Windows without requiring WSL2, making it ideal for this setup.
- CUDA 11.2 and cuDNN 8.1 match TensorFlow 2.10.0's requirements for GPU support.

### Step 5: Verify GPU Support
After installation, test if TensorFlow can detect your GPU:
```bash
python -c "import tensorflow as tf; print('GPU available' if tf.config.list_physical_devices('GPU') else 'No GPU available')"
```
- Expected output: `GPU available`
- If it says `No GPU available`, see the troubleshooting section below.

---

## Running the Project
To train the AI or play the game:
1. Ensure you’re in the `tetris_ai` environment:
   ```bash
   conda activate tetris_ai
   ```
2. Run the main script:
   ```bash
   python ai_trainer_full.py
   ```
   - If no saved model is found, it starts fresh training.
   - Training will use the GPU if detected; otherwise, it falls back to CPU (slower).

---

## Project Structure
- `ai_trainer_full.py`: Main script containing the game logic, AI model, and training loop.
- Other files (to be added as needed): May include model definitions, utilities, or saved weights.

---

## Troubleshooting GPU Issues
If TensorFlow fails to detect your GPU (`No GPU available`), you may see warnings like:
```
Could not load dynamic library 'cufft64_10.dll'; dlerror: cufft64_10.dll not found
Cannot dlopen some GPU libraries...
Skipping registering GPU devices...
```

### Fixes
1. **Ensure CUDA and cuDNN Compatibility**:
   - Verify that `cudatoolkit=11.2` and `cudnn=8.1` are installed in the environment:
     ```bash
     conda list | findstr "cudatoolkit cudnn tensorflow"
     ```
   - If missing, reinstall them:
     ```bash
     conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1
     ```

2. **Check NVIDIA Drivers**:
   - Ensure your NVIDIA GPU drivers are up to date. Download from [NVIDIA Driver Downloads](https://www.nvidia.com/Download/index.aspx).

3. **Reinstall TensorFlow**:
   - If issues persist, uninstall and reinstall TensorFlow:
     ```bash
     pip uninstall tensorflow
     pip install tensorflow==2.10.0
     ```

4. **System CUDA Conflict**:
   - If you have CUDA 12.2 installed system-wide (check with `nvcc --version`), it won’t affect the Conda environment since `cudatoolkit` is isolated. Avoid modifying the system PATH unless necessary.

---

## Notes
- **Performance**: GPU training is significantly faster than CPU. Ensure GPU support is working for optimal performance.
- **Upgrading TensorFlow**: If you prefer a newer TensorFlow version (e.g., 2.13.0 with CUDA 12.0), adjust the `cudatoolkit` version accordingly, but note that versions >=2.11 require WSL2 on Windows for GPU support.
- **Contributing**: Feel free to submit pull requests or issues to improve the project!

---

## License
This project is open-source under the [MIT License](LICENSE.MD).

