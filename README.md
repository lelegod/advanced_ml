# Advanced Machine Learning - DTU

This repository contains notes and exercises for the **Advanced Machine Learning** course at DTU.

**Author:** Kyle Nathan Yahya

## 📂 Project Structure

- `notes/`: LaTeX source files for course notes.
- `week{n}/`: Exercises and solutions for each week.
- `tasks.py`: Automation scripts for environment management.

## 🚀 Setup & Installation

This project uses **Conda** for environment management and **Invoke** for task automation.

### 1. Prerequisites
Ensure you have [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://www.anaconda.com/) installed.

### 2. Create the Environment
To create the conda environment (`advanced_ml`) with Python 3.12 and install base dependencies:

```bash
invoke create-env
```

If you prefer to create the environment in a specific path (local folder):

```bash
invoke create-env --path .
```

### 3. Install Dependencies
To install the required Python packages (including GPU-accelerated PyTorch if available):

```bash
invoke install
```

> **Note:** The script automatically detects if `nvidia-smi` is available and installs the appropriate PyTorch version (CUDA 12.6 for GPU or CPU-only).

## 📝 Usage

### Running Exercises
Activate the environment and launch Jupyter Lab:

```bash
conda activate advanced_ml
jupyter lab
```

### Building Notes
Dependencies for compiling the LaTeX notes can be found in `notes/`.
