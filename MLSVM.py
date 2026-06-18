import sys
import json
import os
import pickle
import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

MODEL_DIR     = os.path.dirname(os.path.abspath(__file__))
MODEL_FILE    = os.path.join(MODEL_DIR, 'svm_model.pkl')
SCALER_FILE   = os.path.join(MODEL_DIR, 'svm_scaler.pkl')
FEATURES_FILE = os.path.join(MODEL_DIR, 'svm_features.pkl')
CSV_FILE      = os.path.join(MODEL_DIR, 'user_typing_data.csv')

NUMERIC_FEATURES = [
    'WPM', 'Accuracy(%)', 'Characters', 'Errors', 'Regressions', 'Time(s)',
    'AvgDwellTime(ms)', 'AvgFlightTime(ms)', 'AvgDDTime(ms)', 'AvgUUTime(ms)',
    'LeftShiftCount', 'RightShiftCount',
    'BackspaceCount', 'CtrlACount', 'CapsLockCount',
    'RolloverCount', 'AvgRolloverDuration(ms)',
    'ErrorRate(%)', 'Burstiness', 'SpeedDecayRate'
]
CATEGORICAL_FEATURES = ['ShiftPreference', 'BackspaceStyle']

SESSION_KEY_MAP = {
    'WPM':                   'wpm',
    'Accuracy(%)':           'accuracy',
    'Characters':            'characters',
    'Errors':                'errors',
    'Regressions':           'regressions',
    'Time(s)':               'time',
    'AvgDwellTime(ms)':      'avgDwellMs',
    'AvgFlightTime(ms)':     'avgFlightMs',
    'AvgDDTime(ms)':         'avgDDMs',
    'AvgUUTime(ms)':         'avgUUMs',
    'LeftShiftCount':        'leftShiftCount',
    'RightShiftCount':       'rightShiftCount',
    'BackspaceCount':        'backspaceCount',
    'CtrlACount':            'ctrlACount',
    'CapsLockCount':         'capsLockCount',
    'RolloverCount':         'rolloverCount',
    'AvgRolloverDuration(ms)': 'avgRolloverDurationMs',
    'ErrorRate(%)':          'errorRate',
    'Burstiness':            'burstiness',
    'SpeedDecayRate':        'speedDecayRate',
}


def load_and_prepare_data(csv_path):
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=['Name'])
    df['Name'] = df['Name'].str.strip()
    df = df[df['Name'] != '']

    for col in NUMERIC_FEATURES:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    X_raw = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES].copy()
    for col in CATEGORICAL_FEATURES:
        X_raw[col] = X_raw[col].fillna('none').str.strip().str.lower()

    y = df['Name']
    X_encoded = pd.get_dummies(X_raw, columns=CATEGORICAL_FEATURES)
    return X_encoded, y, X_encoded.columns.tolist()


def train_model():
    print("Loading data...", file=sys.stderr)
    X, y, feature_cols = load_and_prepare_data(CSV_FILE)

    unique_classes = y.unique().tolist()
    print(f"Classes found: {unique_classes}", file=sys.stderr)
    print(f"Total samples: {len(X)}", file=sys.stderr)

    scaler = StandardScaler()

    if len(unique_classes) == 1:
        print("Single class — training One-Class SVM (novelty detection)...", file=sys.stderr)
        X_scaled = scaler.fit_transform(X)
        model = svm.OneClassSVM(kernel='rbf', nu=0.1, gamma='scale')
        model.fit(X_scaled)
        model_type  = 'oneclass'
        known_class = unique_classes[0]
        accuracy    = None
        print(f"One-Class SVM trained on '{known_class}' samples.", file=sys.stderr)
    else:
        print("Multiple classes — training binary/multi SVC...", file=sys.stderr)
        if len(X) < 5:
            X_train, X_test, y_train, y_test = X, X, y, y
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42,
                stratify=y if all(c >= 2 for c in y.value_counts()) else None
            )
        X_train_s = scaler.fit_transform(X_train)
        X_test_s  = scaler.transform(X_test)
        model = svm.SVC(kernel='rbf', C=1.0, gamma='scale')
        model.fit(X_train_s, y_train)
        y_pred   = model.predict(X_test_s)
        accuracy = round(float(accuracy_score(y_test, y_pred)), 4)
        model_type  = 'svc'
        known_class = None
        print(f"SVC trained. Test accuracy: {accuracy:.2%}", file=sys.stderr)

    with open(MODEL_FILE,    'wb') as f:
        pickle.dump({'model': model, 'type': model_type, 'known_class': known_class}, f)
    with open(SCALER_FILE,   'wb') as f:
        pickle.dump(scaler, f)
    with open(FEATURES_FILE, 'wb') as f:
        pickle.dump(feature_cols, f)

    result = {
        'success':    True,
        'model_type': model_type,
        'classes':    unique_classes,
        'samples':    int(len(X)),
        'accuracy':   accuracy
    }
    print(json.dumps(result))


def build_feature_row(session, feature_cols):
    row = {}
    for csv_col, json_key in SESSION_KEY_MAP.items():
        row[csv_col] = float(session.get(json_key, 0) or 0)

    row_df = pd.DataFrame([row])

    shift_pref    = str(session.get('shiftPreference', 'none') or 'none').strip().lower()
    backspace_sty = str(session.get('backspaceStyle', 'minimal') or 'minimal').strip().lower()

    for col in feature_cols:
        if col not in row_df.columns:
            row_df[col] = 0

    shift_col = f'ShiftPreference_{shift_pref}'
    if shift_col in feature_cols:
        row_df[shift_col] = 1

    bs_col = f'BackspaceStyle_{backspace_sty}'
    if bs_col in feature_cols:
        row_df[bs_col] = 1

    row_df = row_df.reindex(columns=feature_cols, fill_value=0)
    return row_df


def predict(session_json_str):
    if not os.path.exists(MODEL_FILE):
        print(json.dumps({'error': 'Model not trained yet.'}))
        return

    with open(MODEL_FILE,    'rb') as f:
        model_data = pickle.load(f)
    with open(SCALER_FILE,   'rb') as f:
        scaler = pickle.load(f)
    with open(FEATURES_FILE, 'rb') as f:
        feature_cols = pickle.load(f)

    model       = model_data['model']
    model_type  = model_data['type']
    known_class = model_data.get('known_class', 'Joshua')

    session  = json.loads(session_json_str)
    row_df   = build_feature_row(session, feature_cols)
    X_scaled = scaler.transform(row_df)

    if model_type == 'oneclass':
        pred       = model.predict(X_scaled)[0]
        prediction = known_class if pred == 1 else 'Unknown'
        score      = float(model.decision_function(X_scaled)[0])
        confidence = round(score, 3)
    else:
        prediction = str(model.predict(X_scaled)[0])
        try:
            scores     = model.decision_function(X_scaled)[0]
            confidence = round(float(np.max(np.abs(scores if hasattr(scores, '__len__') else [scores]))), 3)
        except Exception:
            confidence = None

    print(json.dumps({
        'prediction':  prediction,
        'confidence':  confidence,
        'model_type':  model_type
    }))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GuessTyper SVM — train or predict')
    parser.add_argument('--train',   action='store_true', help='Train the model on CSV data')
    parser.add_argument('--predict', action='store_true', help='Read session JSON from stdin and classify')
    args = parser.parse_args()

    if args.predict:
        session_json = sys.stdin.read().strip()
        predict(session_json)
    else:
        train_model()
