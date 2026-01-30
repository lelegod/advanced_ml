# Advanced Machine Learning

Notes and solutions for the **Advanced Machine Learning** course.

**Author:** Kyle Nathan Yahya

---

## 🚀 Setup & Installation

### 1. Python Environment
This project manages dependencies using `conda` and `invoke`.

1. **Install Conda** (if not already installed).
2. **Create the environment**:
   ```bash
   invoke create-env
   ```
3. **Install dependencies**:
   ```bash
   invoke install
   ```
   *Note: This automatically installs packages from `requirements.txt`.*

### 2. LaTeX Setup (for Notes)

To edit and compile the LaTeX notes (`notes/main.tex`) in VS Code, follow these steps:

#### **Step 1: Install TeX Live**
You need a TeX distribution to compile `.tex` files.
- **Windows**: Download [TeX Live](https://tug.org/texlive/acquire-netinstall.html) (install-tl-windows.exe).
  - *Tip: The full installation is large (~5GB+). You can choose a smaller scheme if you only need basic functionality, but "Full" is safest.*
- **Mac**: Install [MacTeX](https://tug.org/mactex/).
- **Linux**: `sudo apt install texlive-full` (or equivalent).

#### **Step 2: Install VS Code Extension**
Install the **LaTeX Workshop** extension by James Yu.
- Open VS Code Extensions (`Ctrl+Shift+X`).
- Search for `LaTeX Workshop`.
- Click **Install**.

#### **Step 3: Usage**
- Open `notes/main.tex`.
- The extension should automatically build the PDF on save (configured in `.vscode/settings.json`).
- View the PDF by clicking the "View LaTeX PDF" icon in the top right or pressing `Ctrl+Alt+V`.
- **Formatting**: The project is configured to use `latexindent`. You can format the document using `Shift+Alt+F`.

---

## 📂 Project Structure
- `notes/`: LaTeX notes and weekly summaries.
- `tasks.py`: Automation scripts.
- `requirements.txt`: Python dependencies.