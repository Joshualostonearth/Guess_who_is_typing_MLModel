# GuessTyper

GuessTyper is a dynamic typing speed test application that incorporates **Keystroke Biometric Analysis**. It measures not just how fast you type, but *how* you type, and uses a Machine Learning model (One-Class SVM) to verify if the person typing matches the known user's unique typing signature.

## Features

- **Typing Speed Test**: Standard metrics like WPM, Accuracy, Characters typed, Errors, and Regressions.
- **Keystroke Biometrics**: Tracks microscopic timing patterns:
  - **Dwell Time**: How long a key is held down.
  - **Flight Time**: The time between releasing one key and pressing the next.
  - **Digraph/Trigraph Latencies**: Timing for specific common letter combinations (e.g., "th", "ing").
  - **Behavioral Patterns**: Shift key preference (left vs right), backspace style (held vs tapped), and key rollover counts.
- **Machine Learning Verification**: Uses a `scikit-learn` One-Class Support Vector Machine (SVM) to learn the primary user's typing pattern and flag anomalous sessions as "Unknown".
- **Express.js Backend**: Handles session logging to a CSV dataset and spawning Python ML scripts.

## Tech Stack

- **Frontend**: Vanilla HTML, CSS, JavaScript
- **Backend**: Node.js, Express.js
- **Machine Learning**: Python, `pandas`, `scikit-learn`

## Setup Instructions

### Prerequisites
- Node.js (v14 or higher)
- Python 3.8+ 

### 1. Install Node Dependencies
```bash
npm install
```

### 2. Setup Python Virtual Environment
```bash
python -m venv venv

# On Windows:
venv\Scripts\activate
# On Mac/Linux:
# source venv/bin/activate

# Install required Python packages
pip install pandas scikit-learn
```

### 3. Run the Server
```bash
npm start
```
The application will be available at `http://localhost:3000`.

## How the ML Model Works

The project uses a **One-Class SVM** for novelty detection. 
1. **Data Collection**: When the primary user types, their biometric features are extracted and appended to `user_typing_data.csv`.
2. **Training**: The Python script (`MLSVM.py`) reads the CSV, scales the numeric features, and trains the One-Class SVM exclusively on the known user's data. It saves the model binaries (`.pkl` files).
3. **Prediction**: After completing a typing test, the frontend sends the extracted features to the backend. The Python script evaluates the new session against the trained model. If the pattern deviates significantly from the learned boundaries, it classifies the typer as "Unknown".

## Note on Privacy
The `user_typing_data.csv` contains personal biometric typing signatures. It is deliberately excluded from version control (`.gitignore`) to keep this data private. When cloning this repository, a new empty CSV will be automatically generated upon the first server startup.
