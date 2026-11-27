from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score

ROOT = Path(__file__).parent / "data" / "UCI HAR Dataset"

SIGNALS = [
    "total_acc_x", "total_acc_y", "total_acc_z",
    "body_gyro_x", "body_gyro_y", "body_gyro_z",
]

# UCI HAR (0-indexed) : 0 walking, 1 upstairs, 2 downstairs,
# 3 sitting, 4 standing, 5 laying
# Regroupement : 0 = marche, 1 = escaliers, 2 = immobile
GROUP_MAP = np.array([0, 1, 1, 2, 2, 2])
GROUP_NAMES = ["marche", "escaliers", "immobile"]

FS = 50  # Hz


def load_split(split):
    arrays = []
    for name in SIGNALS:
        path = ROOT / split / "Inertial Signals" / f"{name}_{split}.txt"
        arrays.append(np.loadtxt(path))
    X = np.stack(arrays, axis=-1)  # (n, 128, 6)
    y = np.loadtxt(ROOT / split / f"y_{split}.txt", dtype=int) - 1
    return X, y


def simple_features(X):
    """mean, std, min, max, énergie par axe -> 30 features."""
    return np.concatenate([
        X.mean(axis=1),
        X.std(axis=1),
        X.min(axis=1),
        X.max(axis=1),
        (X ** 2).mean(axis=1),
    ], axis=1)


def rich_features(X):
    """Features simples + magnitude, IQR, corrélations, spectre (FFT)."""
    n = X.shape[0]
    feats = [simple_features(X)]

    # IQR par axe
    q75, q25 = np.percentile(X, [75, 25], axis=1)
    feats.append(q75 - q25)

    # Magnitude de l'accélération et du gyroscope
    acc_mag = np.linalg.norm(X[:, :, 0:3], axis=2)  # (n, 128)
    gyr_mag = np.linalg.norm(X[:, :, 3:6], axis=2)
    for mag in (acc_mag, gyr_mag):
        feats.append(np.stack([
            mag.mean(axis=1), mag.std(axis=1), mag.min(axis=1), mag.max(axis=1),
        ], axis=1))

    # Corrélations entre axes de l'accéléromètre (xy, xz, yz)
    corr = np.zeros((n, 3))
    for i, (a, b) in enumerate([(0, 1), (0, 2), (1, 2)]):
        xa = X[:, :, a] - X[:, :, a].mean(axis=1, keepdims=True)
        xb = X[:, :, b] - X[:, :, b].mean(axis=1, keepdims=True)
        denom = np.sqrt((xa ** 2).sum(axis=1) * (xb ** 2).sum(axis=1)) + 1e-9
        corr[:, i] = (xa * xb).sum(axis=1) / denom
    feats.append(corr)

    # Spectre de la magnitude d'accélération (sans la composante continue)
    spec = np.abs(np.fft.rfft(acc_mag - acc_mag.mean(axis=1, keepdims=True), axis=1))
    freqs = np.fft.rfftfreq(acc_mag.shape[1], d=1 / FS)
    dom_freq = freqs[spec.argmax(axis=1)]
    spec_energy = (spec ** 2).mean(axis=1)
    # Énergie dans 4 bandes : 0-1, 1-2, 2-4, 4-8 Hz
    bands = [(0, 1), (1, 2), (2, 4), (4, 8)]
    band_feats = [
        (spec[:, (freqs >= lo) & (freqs < hi)] ** 2).sum(axis=1) for lo, hi in bands
    ]
    feats.append(np.stack([dom_freq, spec_energy] + band_feats, axis=1))

    return np.concatenate(feats, axis=1)


def run(name, F_train, y_train, F_test, y_test, labels, verbose=False):
    clf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    clf.fit(F_train, y_train)
    pred = clf.predict(F_test)
    f1 = f1_score(y_test, pred, average="macro")
    print(f"{name:<40} features={F_train.shape[1]:<4} F1 macro = {f1:.4f}")
    if verbose:
        print("\nMatrice de confusion :\n", confusion_matrix(y_test, pred))
        print("\n", classification_report(y_test, pred, target_names=labels))
    return f1


def main():
    labels6 = [
        line.split()[1]
        for line in (ROOT / "activity_labels.txt").read_text().splitlines()
    ]

    X_train, y6_train = load_split("train")
    X_test, y6_test = load_split("test")
    print("Train:", X_train.shape, "Test:", X_test.shape, "\n")

    y3_train = GROUP_MAP[y6_train]
    y3_test = GROUP_MAP[y6_test]

    S_train, S_test = simple_features(X_train), simple_features(X_test)
    R_train, R_test = rich_features(X_train), rich_features(X_test)

    run("6 classes, features simples", S_train, y6_train, S_test, y6_test, labels6)
    run("3 classes, features simples", S_train, y3_train, S_test, y3_test, GROUP_NAMES)
    run("6 classes, features enrichies", R_train, y6_train, R_test, y6_test, labels6)
    print()
    print("=== Détail : 3 classes, features enrichies ===")
    run("3 classes, features enrichies", R_train, y3_train, R_test, y3_test,
        GROUP_NAMES, verbose=True)


if __name__ == "__main__":
    main()