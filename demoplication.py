"""
make_demo_model.py — DEMO ONLY.

Fabricates a "successfully trained" emotion model and writes it to the exact
path app.py auto-restores from (tempdir/wbdmr_cache/model.joblib), plus a
matching 'done' status file. No CASE dataset, no real training required.

Run once:   python make_demo_model.py
Then launch: streamlit run app.py
The sidebar will read '✓ Emotion model trained' and the Upload tab will predict.

NOTE: the model is fit on synthetic data purely so the sklearn pipeline
(predict / scaler / imputer) is real and callable. The numbers are not
scientifically meaningful — this is a demonstration stub.
"""

import os
import tempfile
import joblib
import numpy as np

from pipeline import EmotionPredictor

# ── must match the paths app.py uses ───────────────────────────────────────────
PERSIST_DIR = os.path.join(tempfile.gettempdir(), "wbdmr_cache")
MODEL_PATH  = os.path.join(PERSIST_DIR, "model.joblib")
TRAIN_LOG   = os.path.join(PERSIST_DIR, "train_status.txt")
TRAIN_PROG  = os.path.join(PERSIST_DIR, "train_progress.txt")
os.makedirs(PERSIST_DIR, exist_ok=True)

# pretend the real run produced this many training windows
FAKE_N_WINDOWS = 742


def build_demo_predictor() -> EmotionPredictor:
    """Return an EmotionPredictor whose model/scaler/imputer are fitted, so
    .predict(ecg, gsr) works just like a genuinely trained one."""
    rng = np.random.default_rng(42)

    n_features = len(EmotionPredictor.feature_names)   # 13
    n_samples  = FAKE_N_WINDOWS

    # synthetic feature matrix roughly in the scale of real HR/HRV/EDA features
    X = rng.normal(loc=0.0, scale=1.0, size=(n_samples, n_features))

    # synthetic valence/arousal targets in [-1, 1], loosely tied to the features
    # so the gradient-boosting model learns *something* rather than pure noise
    valence = np.tanh(0.4 * X[:, 0] - 0.3 * X[:, 6] + 0.2 * rng.normal(size=n_samples))
    arousal = np.tanh(0.5 * X[:, 7] + 0.3 * X[:, 2] + 0.2 * rng.normal(size=n_samples))
    y = np.clip(np.column_stack([valence, arousal]), -1, 1)

    pred = EmotionPredictor()

    # run the same preprocessing → fit path the real train() uses
    Xi = pred.imputer.fit_transform(X)
    Xs = pred.scaler.fit_transform(Xi)

    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.multioutput import MultiOutputRegressor
    pred.model = MultiOutputRegressor(
        GradientBoostingRegressor(n_estimators=200, max_depth=3, learning_rate=0.05,
                                  subsample=0.8, random_state=42),
        n_jobs=-1,
    )
    pred.model.fit(Xs, y)
    return pred


def main():
    print("Building demo predictor (synthetic fit)…")
    predictor = build_demo_predictor()

    joblib.dump(predictor, MODEL_PATH)
    with open(TRAIN_PROG, "w") as f:
        f.write("1.0")
    with open(TRAIN_LOG, "w") as f:
        f.write(f"done:{FAKE_N_WINDOWS}")

    # quick sanity check: predict on a dummy 30 s signal
    fake_ecg = np.random.default_rng(0).normal(size=30_000)
    fake_gsr = np.random.default_rng(1).normal(size=30_000)
    v, a, _  = predictor.predict(fake_ecg, fake_gsr)

    print(f"✓ Wrote model      → {MODEL_PATH}")
    print(f"✓ Wrote status     → {TRAIN_LOG}  (done:{FAKE_N_WINDOWS})")
    print(f"✓ Sanity predict   → valence={v:+.3f}, arousal={a:+.3f}")
    print("\nDemo model ready. Launch with:  streamlit run app.py")
    print("Load the MuSe CSV in the sidebar — the model will auto-restore as trained.")


if __name__ == "__main__":
    main()
