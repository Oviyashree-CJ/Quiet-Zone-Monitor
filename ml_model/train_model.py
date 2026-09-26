import os
import numpy as np
import librosa
import joblib

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report


# ==============================
# Feature Extraction
# ==============================
def extract_features(file_path):
    try:
        y, sr = librosa.load(file_path, sr=16000)

        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)

        mfcc_mean = np.mean(mfcc.T, axis=0)
        mfcc_std = np.std(mfcc.T, axis=0)

        return np.hstack((mfcc_mean, mfcc_std))

    except Exception as e:
        print("Error processing:", file_path)
        return None


# ==============================
# Load Dataset
# ==============================
dataset_path = os.path.join(os.path.dirname(__file__), "dataset")

X = []
y = []

for label in ["human", "non_human"]:
    folder = os.path.join(dataset_path, label)

    print(f"\nLoading {label} data...")

    for root, dirs, files in os.walk(folder):
        for file in files:
            if file.lower().endswith(".wav"):
                file_path = os.path.join(root, file)

                features = extract_features(file_path)

                if features is not None:
                    X.append(features)
                    y.append(label)

print("\nTotal samples:", len(X))


# ==============================
# Safety Check
# ==============================
if len(X) == 0:
    print("❌ No audio files found. Check dataset path!")
    exit()


# ==============================
# Convert to numpy
# ==============================
X = np.array(X)
y = np.array(y)


# ==============================
# Feature Scaling
# ==============================
scaler = StandardScaler()
X = scaler.fit_transform(X)

joblib.dump(scaler, "ml_model/scaler.pkl")


# ==============================
# Train/Test Split
# ==============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# ==============================
# Train Model (SVM)
# ==============================
model = SVC(kernel="rbf", probability=True)
model.fit(X_train, y_train)


# ==============================
# Evaluate
# ==============================
y_pred = model.predict(X_test)

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))


# ==============================
# Save Model
# ==============================
joblib.dump(model, "ml_model/human_sound_model.pkl")

print("\n✅ Model trained and saved successfully!")