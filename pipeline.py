import os
import numpy as np
import pandas as pd
import neurokit2 as nk
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from typing import Optional, Callable
import warnings
warnings.filterwarnings('ignore')


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SAMPLING_RATE   = 1000          # Hz
WINDOW_SEC      = 30            # seconds per feature window
WINDOW_SAMPLES  = WINDOW_SEC * SAMPLING_RATE
STEP_SEC        = 10            # hop between windows (seconds)
STEP_SAMPLES    = STEP_SEC * SAMPLING_RATE
MIN_WINDOWS     = 1             # keep subjects even if only 1 window extracted


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------
class EmotionPredictor:
    """
    Extracts physiological features from ECG + EDA windows and trains
    a regression model to predict valence and arousal.

    Key design decisions vs. the original:
    - Sliding-window extraction: each 30-second window becomes one training
      sample (with 10-second hop), giving 10-50x more data per subject.
    - Global StandardScaler fit on ALL training windows — no more per-subject
      self-subtraction that zeroed out every feature vector.
    - GradientBoosting instead of RandomForest: handles small-to-medium
      datasets better and is less prone to overfitting.
    - SimpleImputer before scaling: neurokit2 occasionally returns NaN for
      degenerate windows; we fill with the column median rather than crashing.
    """

    FEATURE_NAMES = [
        'HR_mean', 'HR_std',
        'HRV_SDNN', 'HRV_RMSSD', 'HRV_pNN50', 'HRV_MeanNN',
        'EDA_mean', 'EDA_std',
        'EDA_phasic_mean', 'EDA_phasic_std',
        'EDA_tonic_mean',
        'EDA_peaks', 'EDA_peak_rate',
    ]

    def __init__(self):
        self.model        = None
        self.scaler       = StandardScaler()
        self.imputer      = SimpleImputer(strategy='median')
        self.subject_stats: dict = {}   # kept for API compatibility; no longer used for normalization

    # ------------------------------------------------------------------
    # Feature extraction (single window)
    # ------------------------------------------------------------------
    def extract_features(self, ecg_data: np.ndarray, eda_data: np.ndarray,
                         sampling_rate: int = SAMPLING_RATE) -> dict:
        """
        Extract a feature dict from one window of ECG and EDA signals.
        Returns zeros for any feature that fails (degenerate window).
        """
        features = {k: 0.0 for k in self.FEATURE_NAMES}
        try:
            # --- ECG ---
            ecg_clean = nk.ecg_clean(ecg_data, sampling_rate=sampling_rate)
            ecg_sig, _ = nk.ecg_process(ecg_clean, sampling_rate=sampling_rate)
            hrv = nk.hrv_time(ecg_sig, sampling_rate=sampling_rate, show=False)

            features['HR_mean']    = float(ecg_sig["ECG_Rate"].mean())
            features['HR_std']     = float(ecg_sig["ECG_Rate"].std())
            features['HRV_SDNN']   = float(hrv["HRV_SDNN"].values[0])   if "HRV_SDNN"   in hrv.columns else 0.0
            features['HRV_RMSSD']  = float(hrv["HRV_RMSSD"].values[0])  if "HRV_RMSSD"  in hrv.columns else 0.0
            features['HRV_pNN50']  = float(hrv["HRV_pNN50"].values[0])  if "HRV_pNN50"  in hrv.columns else 0.0
            features['HRV_MeanNN'] = float(hrv["HRV_MeanNN"].values[0]) if "HRV_MeanNN" in hrv.columns else 0.0
        except Exception:
            pass   # leave ECG features as 0

        try:
            # --- EDA ---
            eda_sig, eda_info = nk.eda_process(eda_data, sampling_rate=sampling_rate)

            features['EDA_mean']        = float(np.mean(eda_data))
            features['EDA_std']         = float(np.std(eda_data))
            features['EDA_phasic_mean'] = float(eda_sig["EDA_Phasic"].mean())
            features['EDA_phasic_std']  = float(eda_sig["EDA_Phasic"].std())
            features['EDA_tonic_mean']  = float(eda_sig["EDA_Tonic"].mean())

            n_peaks = len(eda_info.get("SCR_Peaks", []))
            features['EDA_peaks']       = float(n_peaks)
            features['EDA_peak_rate']   = float(n_peaks / (len(eda_data) / sampling_rate))
        except Exception:
            pass   # leave EDA features as 0

        return features

    # ------------------------------------------------------------------
    # Windowed feature extraction for one subject
    # ------------------------------------------------------------------
    def _extract_windows(self, ecg: np.ndarray, eda: np.ndarray,
                         valence_series: pd.Series, arousal_series: pd.Series):
        """
        Slide a window over the full recording and extract (features, label)
        pairs.  The label for each window is the mean valence/arousal over the
        corresponding annotation samples.

        Returns
        -------
        X : list of feature arrays
        y : list of [valence, arousal] pairs
        """
        n = min(len(ecg), len(eda))
        X, y = [], []

        starts = range(0, n - WINDOW_SAMPLES + 1, STEP_SAMPLES)
        for start in starts:
            end = start + WINDOW_SAMPLES

            feats = self.extract_features(ecg[start:end], eda[start:end])
            feat_arr = np.array([feats[k] for k in self.FEATURE_NAMES], dtype=float)

            # Map window to annotation indices proportionally
            ann_len   = len(valence_series)
            ann_start = int(start / n * ann_len)
            ann_end   = int(end   / n * ann_len)
            ann_end   = max(ann_end, ann_start + 1)   # at least 1 sample

            v = float(valence_series.iloc[ann_start:ann_end].mean())
            a = float(arousal_series.iloc[ann_start:ann_end].mean())

            X.append(feat_arr)
            y.append([v, a])

        return X, y

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def train_with_subject_normalization(self, physio_dir: str, annot_dir: str,
                                         num_subjects: int = 30,
                                         progress_cb: Optional[Callable] = None) -> int:
        """
        Build a dataset from all subjects using sliding windows, fit a global
        scaler, then train the regression model.
        """
        all_X, all_y = [], []

        loaded = 0
        for sub in range(1, num_subjects + 1):
            try:
                df_physio = pd.read_csv(os.path.join(physio_dir, f"sub_{sub}.csv"))
                df_annot  = pd.read_csv(os.path.join(annot_dir,  f"sub_{sub}.csv"))

                ecg = df_physio["ecg"].values.astype(float)
                eda = df_physio["gsr"].values.astype(float)

                # Validate annotation columns
                val_col = _find_col(df_annot, ['valence', 'Valence'])
                aro_col = _find_col(df_annot, ['arousal', 'Arousal'])
                if val_col is None or aro_col is None:
                    continue

                X_sub, y_sub = self._extract_windows(
                    ecg, eda, df_annot[val_col], df_annot[aro_col]
                )

                if len(X_sub) >= MIN_WINDOWS:
                    all_X.extend(X_sub)
                    all_y.extend(y_sub)
                    loaded += 1

            except Exception:
                pass

            if progress_cb:
                progress_cb(sub / num_subjects * 0.8)   # 80 % for loading

        if len(all_X) < 5:
            raise ValueError(
                f"Only {len(all_X)} training windows extracted from {loaded} subjects. "
                "Check that physio and annotation CSVs match and contain ecg/gsr/valence/arousal columns."
            )

        X = np.array(all_X, dtype=float)
        y = np.array(all_y, dtype=float)

        # --- Impute then scale globally (fixes zero-normalization bug) ---
        X = self.imputer.fit_transform(X)
        X = self.scaler.fit_transform(X)

        # --- Clip labels to [-1, 1] in case annotations are [0,9] scale ---
        if y.max() > 1.5:
            y = (y - y.mean(axis=0)) / (y.std(axis=0) + 1e-8)
            y = np.clip(y, -1, 1)

        # --- Model: GradientBoosting with conservative depth ---
        base = GradientBoostingRegressor(
            n_estimators=200,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.8,
            random_state=42,
        )
        self.model = MultiOutputRegressor(base, n_jobs=-1)
        self.model.fit(X, y)

        if progress_cb:
            progress_cb(1.0)

        return len(X)

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------
    def predict_emotion(self, ecg_data: np.ndarray, eda_data: np.ndarray,
                        subject_id: Optional[int] = None):
        """
        Predict valence and arousal from a raw ECG + EDA recording.

        Returns
        -------
        valence  : float in [-1, 1]
        arousal  : float in [-1, 1]
        features : dict of extracted feature values (for display)
        """
        if self.model is None:
            raise ValueError("Model not trained yet.")

        features   = self.extract_features(ecg_data, eda_data)
        feat_arr   = np.array([[features[k] for k in self.FEATURE_NAMES]], dtype=float)
        feat_arr   = self.imputer.transform(feat_arr)
        feat_arr   = self.scaler.transform(feat_arr)

        prediction = self.model.predict(feat_arr)[0]
        valence    = float(np.clip(prediction[0], -1, 1))
        arousal    = float(np.clip(prediction[1], -1, 1))
        return valence, arousal, features


# ---------------------------------------------------------------------------
# Music Recommender
# ---------------------------------------------------------------------------
class MusicRecommender:
    """
    Recommends songs from the MuSe dataset by nearest-neighbour search in
    valence–arousal space.

    Key fix vs. original: playlist entries are now serialised as plain dicts
    so that song.get('valence', 0) works correctly in the Streamlit UI.
    """

    def __init__(self, database_path: str):
        self.database = pd.read_csv(database_path)
        self._standardize_columns()
        self._normalize_va_range()

    def _standardize_columns(self):
        rename_map = {
            'valence_tags':   'valence',
            'arousal_tags':   'arousal',
            'dominance_tags': 'dominance',
        }
        self.database = self.database.rename(columns=rename_map)
        missing = [c for c in ['valence', 'arousal'] if c not in self.database.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

    def _normalize_va_range(self):
        """Remap [0, 1] → [−1, 1] if the dataset uses that convention."""
        for col in ('valence', 'arousal'):
            col_min = self.database[col].min()
            col_max = self.database[col].max()
            if col_min >= 0 and col_max <= 1:
                self.database[col] = self.database[col] * 2 - 1
            elif col_max > 1:
                # Assume [1, 9] GEMS/SAM scale → normalise to [-1, 1]
                self.database[col] = (self.database[col] - 5) / 4

    def get_genres(self):
        if 'genre' in self.database.columns:
            return sorted(self.database['genre'].dropna().unique().tolist())
        return []

    def get_emotion_label(self, valence: float, arousal: float):
        if valence >= 0 and arousal >= 0:
            return 'Q1', 'Happy / Excited'
        elif valence < 0 and arousal >= 0:
            return 'Q2', 'Angry / Tense'
        elif valence < 0 and arousal < 0:
            return 'Q3', 'Sad / Depressed'
        else:
            return 'Q4', 'Calm / Relaxed'

    def search_by_emotion(self, target_valence: float, target_arousal: float,
                          genre: Optional[str] = None, num_songs: int = 5,
                          exclude_indices: Optional[set] = None) -> pd.DataFrame:
        if genre and 'genre' in self.database.columns:
            candidates = self.database[
                self.database['genre'].str.lower().str.contains(genre.lower(), na=False)
            ].copy()
            if candidates.empty:
                candidates = self.database.copy()
        else:
            candidates = self.database.copy()

        if exclude_indices:
            candidates = candidates[~candidates.index.isin(exclude_indices)]

        if candidates.empty:
            return candidates

        candidates['distance'] = np.sqrt(
            (candidates['valence'] - target_valence) ** 2 +
            (candidates['arousal'] - target_arousal) ** 2
        )
        return candidates.nsmallest(num_songs, 'distance')

    def recommend_playlist(self, current_valence: float, current_arousal: float,
                           target_valence: float, target_arousal: float,
                           genre: Optional[str] = None,
                           playlist_length: int = 5,
                           gradual: bool = True) -> dict:
        current_quad, current_desc = self.get_emotion_label(current_valence, current_arousal)
        target_quad,  target_desc  = self.get_emotion_label(target_valence,  target_arousal)

        playlist     = []
        used_indices = set()

        if gradual and playlist_length >= 3:
            v_steps = np.linspace(current_valence, target_valence, playlist_length + 1)[1:]
            a_steps = np.linspace(current_arousal, target_arousal, playlist_length + 1)[1:]

            for v, a in zip(v_steps, a_steps):
                candidates = self.search_by_emotion(
                    v, a, genre=genre, num_songs=50, exclude_indices=used_indices
                )
                if candidates.empty:
                    continue
                best = candidates.iloc[0]
                # ── FIX: serialise to plain dict so .get() works in the UI ──
                playlist.append(_series_to_song_dict(best))
                used_indices.add(best.name)
        else:
            songs = self.search_by_emotion(
                target_valence, target_arousal, genre=genre,
                num_songs=playlist_length * 3, exclude_indices=used_indices
            )
            for _, song in songs.iterrows():
                if song.name not in used_indices:
                    playlist.append(_series_to_song_dict(song))
                    used_indices.add(song.name)
                if len(playlist) >= playlist_length:
                    break

        return {
            'current_emotion': {
                'valence': current_valence, 'arousal': current_arousal,
                'quadrant': current_quad,   'description': current_desc,
            },
            'target_emotion': {
                'valence': target_valence, 'arousal': target_arousal,
                'quadrant': target_quad,   'description': target_desc,
            },
            'genre': genre,
            'transition_type': 'gradual' if gradual else 'direct',
            'playlist': playlist[:playlist_length],
        }


# ---------------------------------------------------------------------------
# Top-level system
# ---------------------------------------------------------------------------
class EmotionMusicSystem:
    def __init__(self, music_database_path: str):
        self.predictor   = EmotionPredictor()
        self.recommender = MusicRecommender(music_database_path)
        self.trained     = False

    def train_model(self, physio_dir: str, annot_dir: str,
                    progress_cb: Optional[Callable] = None) -> int:
        n = self.predictor.train_with_subject_normalization(
            physio_dir, annot_dir, progress_cb=progress_cb
        )
        self.trained = True
        return n

    def process_and_recommend(self, ecg_data: np.ndarray, eda_data: np.ndarray,
                               target_valence: float, target_arousal: float,
                               genre: Optional[str] = None,
                               subject_id: Optional[int] = None) -> dict:
        if not self.trained:
            raise ValueError("Model not trained yet.")
        valence, arousal, features = self.predictor.predict_emotion(
            ecg_data, eda_data, subject_id
        )
        result = self.recommender.recommend_playlist(
            valence, arousal, target_valence, target_arousal,
            genre=genre, playlist_length=5, gradual=True
        )
        result['extracted_features'] = features
        return result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _find_col(df: pd.DataFrame, candidates: list) -> Optional[str]:
    """Return the first column name from candidates that exists in df."""
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _series_to_song_dict(series: pd.Series) -> dict:
    """
    Convert a pandas Series (one database row) to a plain dict with
    guaranteed 'valence', 'arousal', and 'distance' keys.
    NaN values are replaced with 0 so the UI never receives NaN.
    """
    d = {k: (0.0 if (isinstance(v, float) and np.isnan(v)) else v)
         for k, v in series.items()}
    d.setdefault('valence',  0.0)
    d.setdefault('arousal',  0.0)
    d.setdefault('distance', 0.0)
    return d


def generate_dummy_signals(n: int = 10_000):
    """Synthetic ECG + EDA for quick smoke-testing."""
    t   = np.linspace(0, n / SAMPLING_RATE, n)
    ecg = np.sin(2 * np.pi * 1.2 * t) + np.random.normal(0, 0.1, n)
    eda = 2 + 0.5 * np.sin(2 * np.pi * 0.05 * t) + np.random.normal(0, 0.05, n)
    return ecg, eda
