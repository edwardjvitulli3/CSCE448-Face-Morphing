# CSCE448-Face-Morphing
A collaborative repository containing all of the code and documentation for our Texas A&amp;M computational photography final project: Face Morphing.

# Requirements
- Python 3.8+
- pip (Python package manager)
- C++ build tools (for dlib)
- CMake

## Step 1: Install System Dependencies

### Windows
1. Install [CMake](https://cmake.org/download/). During installation, check "Add to system PATH".
2. Install [Visual Studio Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/) with the C++ workload.

### Ubuntu / Debian
```bash
sudo apt update
sudo apt install cmake build-essential python3-dev libgtk-3-dev
```

### macOS
```bash
brew install cmake
xcode-select --install
```

## Step 2: Install Python Libraries

```bash
pip install numpy matplotlib scipy PySimpleGUI pillow imageio opencv-python dlib
```

If you get errors with `dlib`, install it from source:

```bash
pip install cmake
pip install dlib
```

Or use a precompiled wheel from [https://www.lfd.uci.edu/~gohlke/pythonlibs/#dlib](https://www.lfd.uci.edu/~gohlke/pythonlibs/#dlib) (Windows only).
