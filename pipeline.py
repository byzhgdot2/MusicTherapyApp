import os
import numpy as np
import pandas as pd
import neurokit2 as nk
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from typing import Optional
import warnings
warnings.filterwarnings('ignore')


SAMPLING_RATE  = 1000
WINDOW_SAMPLES = 30 * SAMPLING_RATE
STEP_SAMPLES   = 10 * SAMPLING_RATE


class EmotionPredictor:

    feature_names = [
        'HR_mean', 'HR_std',
        'HRV_SDNN', 'HRV_RMSSD', 'HRV_pNN50', 'HRV_MeanNN',
        'EDA_mean', 'EDA_std',
        'EDA_phasic_mean', 'EDA_phasic_std',
        'EDA_tonic_mean',
        'EDA_peaks', 'EDA_peak_rate',
    ]

    def __init__(self):
        self.model   = None
        self.scaler  = StandardScaler()
        self.imputer = SimpleImputer(strategy='median')

    def extract_features(self, ecg_data, eda_data, sampling_rate=SAMPLING_RATE):
        feats = {k: 0.0 for k in self.feature_names}
        try:
            ecg_clean  = nk.ecg_clean(ecg_data, sampling_rate=sampling_rate)
            ecg_sig, _ = nk.ecg_process(ecg_clean, sampling_rate=sampling_rate)
            hrv        = nk.hrv_time(ecg_sig, sampling_rate=sampling_rate, show=False)

            feats['HR_mean']    = float(ecg_sig["ECG_Rate"].mean())
            feats['HR_std']     = float(ecg_sig["ECG_Rate"].std())
            feats['HRV_SDNN']   = float(hrv["HRV_SDNN"].values[0])   if "HRV_SDNN"   in hrv.columns else 0.0
            feats['HRV_RMSSD']  = float(hrv["HRV_RMSSD"].values[0])  if "HRV_RMSSD"  in hrv.columns else 0.0
            feats['HRV_pNN50']  = float(hrv["HRV_pNN50"].values[0])  if "HRV_pNN50"  in hrv.columns else 0.0
            feats['HRV_MeanNN'] = float(hrv["HRV_MeanNN"].values[0]) if "HRV_MeanNN" in hrv.columns else 0.0
        except Exception:
            pass

        try:
            eda_sig, eda_info = nk.eda_process(eda_data, sampling_rate=sampling_rate)

            feats['EDA_mean']        = float(np.mean(eda_data))
            feats['EDA_std']         = float(np.std(eda_data))
            feats['EDA_phasic_mean'] = float(eda_sig["EDA_Phasic"].mean())
            feats['EDA_phasic_std']  = float(eda_sig["EDA_Phasic"].std())
            feats['EDA_tonic_mean']  = float(eda_sig["EDA_Tonic"].mean())

            n_peaks             = len(eda_info.get("SCR_Peaks", []))
            feats['EDA_peaks']      = float(n_peaks)
            feats['EDA_peak_rate']  = float(n_peaks / (len(eda_data) / sampling_rate))
        except Exception:
            pass

        return feats

    def _get_windows(self, ecg, eda, valence_col, arousal_col):
        n = min(len(ecg), len(eda))
        X, y = [], []

        for start in range(0, n - WINDOW_SAMPLES + 1, STEP_SAMPLES):
            end      = start + WINDOW_SAMPLES
            feats    = self.extract_features(ecg[start:end], eda[start:end])
            feat_arr = np.array([feats[k] for k in self.feature_names], dtype=float)

            ann_len   = len(valence_col)
            ann_start = int(start / n * ann_len)
            ann_end   = max(int(end / n * ann_len), ann_start + 1)

            v = float(valence_col.iloc[ann_start:ann_end].mean())
            a = float(arousal_col.iloc[ann_start:ann_end].mean())

            X.append(feat_arr)
            y.append([v, a])

        return X, y

    def train(self, physio_dir, annot_dir, num_subjects=30, progress_cb=None):
        all_X, all_y = [], []

        for sub in range(1, num_subjects + 1):
            try:
                physio = pd.read_csv(os.path.join(physio_dir, f"sub_{sub}.csv"))
                annot  = pd.read_csv(os.path.join(annot_dir,  f"sub_{sub}.csv"))

                ecg = physio["ecg"].values.astype(float)
                eda = physio["gsr"].values.astype(float)

                val_col = next((c for c in ['valence', 'Valence'] if c in annot.columns), None)
                aro_col = next((c for c in ['arousal', 'Arousal'] if c in annot.columns), None)
                if val_col is None or aro_col is None:
                    continue

                X_sub, y_sub = self._get_windows(ecg, eda, annot[val_col], annot[aro_col])
                all_X.extend(X_sub)
                all_y.extend(y_sub)

            except Exception:
                pass

            if progress_cb:
                progress_cb(sub / num_subjects * 0.8)

        if len(all_X) < 5:
            raise ValueError(f"Not enough training data — only {len(all_X)} windows extracted.")

        X = self.imputer.fit_transform(np.array(all_X, dtype=float))
        X = self.scaler.fit_transform(X)
        y = np.array(all_y, dtype=float)

        if y.max() > 1.5:
            y = np.clip((y - y.mean(axis=0)) / (y.std(axis=0) + 1e-8), -1, 1)

        self.model = MultiOutputRegressor(
            GradientBoostingRegressor(n_estimators=200, max_depth=3, learning_rate=0.05,
                                      subsample=0.8, random_state=42),
            n_jobs=-1
        )
        self.model.fit(X, y)

        if progress_cb:
            progress_cb(1.0)

        return len(X)

    def predict(self, ecg_data, eda_data):
        if self.model is None:
            raise ValueError("Model not trained yet.")

        feats    = self.extract_features(ecg_data, eda_data)
        feat_arr = np.array([[feats[k] for k in self.feature_names]], dtype=float)
        feat_arr = self.scaler.transform(self.imputer.transform(feat_arr))

        pred    = self.model.predict(feat_arr)[0]
        valence = float(np.clip(pred[0], -1, 1))
        arousal = float(np.clip(pred[1], -1, 1))
        return valence, arousal, feats


class MusicRecommender:

    def __init__(self, database_path):
        self.db = pd.read_csv(database_path).rename(columns={
            'valence_tags': 'valence',
            'arousal_tags': 'arousal',
            'dominance_tags': 'dominance',
        })
        missing = [c for c in ['valence', 'arousal'] if c not in self.db.columns]
        if missing:
            raise ValueError(f"Missing columns: {missing}")
        self._normalize()

    def _normalize(self):
        for col in ('valence', 'arousal'):
            mn, mx = self.db[col].min(), self.db[col].max()
            if mn >= 0 and mx <= 1:
                self.db[col] = self.db[col] * 2 - 1
            elif mx > 1:
                self.db[col] = (self.db[col] - 5) / 4

    def get_genres(self):
        if 'genre' in self.db.columns:
            return sorted(self.db['genre'].dropna().unique().tolist())
        return []

    def emotion_label(self, valence, arousal):
        if valence >= 0 and arousal >= 0:
            return 'Q1', 'Happy / Excited'
        elif valence < 0 and arousal >= 0:
            return 'Q2', 'Angry / Tense'
        elif valence < 0 and arousal < 0:
            return 'Q3', 'Sad / Depressed'
        else:
            return 'Q4', 'Calm / Relaxed'

    def search(self, target_v, target_a, genre=None, n=5, exclude=None):
        if genre and 'genre' in self.db.columns:
            pool = self.db[self.db['genre'].str.lower().str.contains(genre.lower(), na=False)].copy()
            if pool.empty:
                pool = self.db.copy()
        else:
            pool = self.db.copy()

        if exclude:
            pool = pool[~pool.index.isin(exclude)]

        if pool.empty:
            return pool

        pool['distance'] = np.sqrt((pool['valence'] - target_v) ** 2 + (pool['arousal'] - target_a) ** 2)
        return pool.nsmallest(n, 'distance')

    def recommend(self, curr_v, curr_a, target_v, target_a, genre=None, length=5, gradual=True):
        curr_quad, curr_desc   = self.emotion_label(curr_v, curr_a)
        target_quad, target_desc = self.emotion_label(target_v, target_a)

        playlist, seen = [], set()

        if gradual and length >= 3:
            for v, a in zip(np.linspace(curr_v, target_v, length + 1)[1:],
                            np.linspace(curr_a, target_a, length + 1)[1:]):
                hits = self.search(v, a, genre=genre, n=50, exclude=seen)
                if hits.empty:
                    continue
                best = hits.iloc[0]
                playlist.append(_to_dict(best))
                seen.add(best.name)
        else:
            for _, song in self.search(target_v, target_a, genre=genre, n=length * 3, exclude=seen).iterrows():
                playlist.append(_to_dict(song))
                seen.add(song.name)
                if len(playlist) >= length:
                    break

        return {
            'current_emotion':  {'valence': curr_v,   'arousal': curr_a,   'quadrant': curr_quad,   'description': curr_desc},
            'target_emotion':   {'valence': target_v,  'arousal': target_a,  'quadrant': target_quad,  'description': target_desc},
            'playlist': playlist[:length],
        }


class EmotionMusicSystem:

    def __init__(self, music_database_path):
        self.predictor   = EmotionPredictor()
        self.recommender = MusicRecommender(music_database_path)
        self.trained     = False

    def train_model(self, physio_dir, annot_dir, progress_cb=None):
        n = self.predictor.train(physio_dir, annot_dir, progress_cb=progress_cb)
        self.trained = True
        return n

    def predict_emotion(self, ecg_data, eda_data):
        return self.predictor.predict(ecg_data, eda_data)

    def get_emotion_label(self, valence, arousal):
        return self.recommender.emotion_label(valence, arousal)


def _to_dict(series):
    d = {k: (0.0 if isinstance(v, float) and np.isnan(v) else v) for k, v in series.items()}
    d.setdefault('valence', 0.0)
    d.setdefault('arousal', 0.0)
    d.setdefault('distance', 0.0)
    return d
