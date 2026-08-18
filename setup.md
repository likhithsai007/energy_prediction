# Installation and Setup Guide

This guide provides instructions to set up the environment and install dependencies to run the Appliances Energy Prediction data mining pipeline.

## 1. Prerequisites
Ensure you have the following installed on your system:
- **Python 3.8 or higher**
- **pip** (Python package installer)
- **Git** (for version control)

---

## 2. Environment Setup

It is highly recommended to use a virtual environment to manage project-specific dependencies and avoid conflicts with other system-wide packages.

### Option A: Using `venv` (Standard Python)
1. **Open your terminal/command prompt** and navigate to the project directory:
   ```bash
   cd path/to/Appliances-energy-prediction-data
   ```
2. **Create the virtual environment**:
   ```bash
   # Windows
   python -m venv venv

   # macOS/Linux
   python3 -m venv venv
   ```
3. **Activate the virtual environment**:
   ```bash
   # Windows (Command Prompt)
   venv\Scripts\activate.bat

   # Windows (PowerShell)
   .\venv\Scripts\activate.ps1

   # macOS/Linux
   source venv/bin/activate
   ```

### Option B: Using `Conda` (Anaconda/Miniconda)
1. **Create the conda environment**:
   ```bash
   conda create --name energy_prediction python=3.9 -y
   ```
2. **Activate the environment**:
   ```bash
   conda activate energy_prediction
   ```

---

## 3. Install Dependencies

Install all required Python packages from [requirements.txt](file:///c:/Users/likhi/Desktop/Appliances-energy-prediction-data/requirements.txt):

```bash
pip install -r requirements.txt
```

### Dependencies Breakdown:
- **Data Manipulation**: `pandas`, `numpy`
- **Machine Learning**: `scikit-learn`, `joblib`
- **Feature Selection**: `boruta`
- **Data Visualization**: `matplotlib`, `seaborn`
- **Interactive Notebooks**: `ipykernel`, `notebook`

---

## 4. Verification

To verify that the installation was successful, check that all packages are available by running a test import:

```bash
python -c "import pandas, numpy, sklearn, matplotlib, seaborn; print('All packages imported successfully!')"
```

Once verified, you are ready to run the pipeline. Proceed to [guide.md](file:///c:/Users/likhi/Desktop/Appliances-energy-prediction-data/guide.md) for execution details.
