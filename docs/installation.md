# Installation Manual

This document provides step-by-step setup guides for local system deployments.

---

## 1. System Requirements

*   **Operating Systems**: Windows 10/11, macOS (Intel/M-series), or Ubuntu (20.04 or newer).
*   **Python**: Python 3.12 (specifically tested on 3.12.10).
*   **Hardware**: 8 GB RAM minimum (16 GB recommended for large dataset tuning runs).

---

## 2. OS-Specific Shared Library Dependencies

The reporting engine uses **WeasyPrint** for converting HTML templates to executive PDF files, which requires underlying graphic layout engines (Cairo and Pango) to be installed on your OS.

### Ubuntu/Debian
Install the required libraries using `apt-get`:
```bash
sudo apt-get update
sudo apt-get install -y \
    build-essential \
    python3-dev \
    libffi-dev \
    libssl-dev \
    libcairo2 \
    libpango-1.0-0 \
    libpangocairo-1.0-0 \
    libgdk-pixbuf2.0-0 \
    shared-mime-info \
    fonts-dejavu \
    fonts-liberation
```

### macOS
Install the packages using Homebrew:
```bash
brew install cango pango libffi glib
```

### Windows
1.  Download the GTK installer or binaries for Windows (e.g. from MSYS2 or WeasyPrint documentation).
2.  Alternatively, the WeasyPrint package on Windows can locate the libraries automatically if GTK3 is installed and added to the System Path (`%PATH%`).
3.  Or simply deploy using our **Docker container** to completely bypass local OS library installations!

---

## 3. Local Installation Steps

### 1. Clone the Repository
```bash
git clone https://github.com/example/capstone-multi-agent-analyst.git
cd capstone-multi-agent-analyst
```

### 2. Configure Environment variables
Copy the `.env.example` file and fill in your API credentials:
```bash
cp .env.example .env
```

### 3. Create a Virtual Environment & Install Packages
```bash
# Create python venv
python -m venv .venv

# Activate venv
# On macOS / Linux:
source .venv/bin/activate
# On Windows (Command Prompt):
.venv\Scripts\activate.bat
# On Windows (PowerShell):
.venv\Scripts\Activate.ps1

# Upgrade installer and setup dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Verify Local Installation
Execute the quick verification script (pytest) to make sure everything compiles:
```bash
python -m pytest tests/test_utils.py
```
If the tests pass, your system is successfully configured!
