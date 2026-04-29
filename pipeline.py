import os
import numpy as np
import pandas as pd
import neurokit2 as nk
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from typing import Optional, Dict
import warnings
warnings.filterwarnings('ignore')


class EmotionPredictor:
    def __init__(self):
        self.model = None
        self.subject_stats = {}
        self.feature_names = [
            'HR_mean', 'HR_std', 'HRV_SDNN', 'HRV_RMSSD', 'HRV_pNN50', 'HRV_MeanNN',
            'EDA_mean', 'EDA_std', 'EDA_phasic', 'EDA_phasic_std', 'EDA_tonic',
            'EDA_peaks', 'EDA_peak_rate'
        ]

    def extract_features(self, ecg_data, eda_data, sampling_rate=1000):
        features = {}
        try:
            ecg_cleaned = nk.ecg_clean(ecg_data, sampling_rate=sampling_rate)
            signals, info = nk.ecg_process(ecg_cleaned, sampling_rate=sampling_rate)
            hrv = nk.hrv_time(signals, sampling_rate=sampling_rate, show=False)

            features['HR_mean'] = signals["ECG_Rate"].mean()
            features['HR_std'] = signals["ECG_Rate"].std()
            features['HRV_SDNN'] = hrv["HRV_SDNN"].values[0] if "HRV_SDNN" in hrv.columns else 0
            features['HRV_RMSSD'] = hrv["HRV_RMSSD"].values[0] if "HRV_RMSSD" in hrv.columns else 0
            features['HRV_pNN50'] = hrv["HRV_pNN50"].values[0] if "HRV_pNN50" in hrv.columns else 0
            features['HRV_MeanNN'] = hrv["HRV_MeanNN"].values[0] if "HRV_MeanNN" in hrv.columns else 0

            eda_signals, eda_info = nk.eda_process(eda_data, sampling_rate=sampling_rate)
            features['EDA_mean'] = eda_data.mean()
            features['EDA_std'] = eda_data.std()
            features['EDA_phasic'] = eda_signals["EDA_Phasic"].mean()
            features['EDA_phasic_std'] = eda_signals["EDA_Phasic"].std()
            features['EDA_tonic'] = eda_signals["EDA_Tonic"].mean()
            features['EDA_peaks'] = len(eda_info["SCR_Peaks"])
            features['EDA_peak_rate'] = len(eda_info["SCR_Peaks"]) / (len(eda_data) / sampling_rate)
        except Exception:
            for key in self.feature_names:
                features[key] = 0
        return features

    def train_with_subject_normalization(self, physio_dir, annot_dir, num_subjects=30, progress_cb=None):
        all_subjects_data = {}
        for sub in range(1, num_subjects + 1):
            try:
                df_physio = pd.read_csv(os.path.join(physio_dir, f"sub_{sub}.csv"))
                df_annot  = pd.read_csv(os.path.join(annot_dir,  f"sub_{sub}.csv"))
                features  = self.extract_features(df_physio["ecg"].values, df_physio["gsr"].values)
                valence   = df_annot["valence"].mean()
                arousal   = df_annot["arousal"].mean()
                all_subjects_data[sub] = {
                    'features': np.array(list(features.values())),
                    'valence': valence,
                    'arousal': arousal
                }
            except Exception:
                continue
            if progress_cb:
                progress_cb(sub / num_subjects)

        all_features = np.array([data['features'] for data in all_subjects_data.values()])
        global_std   = np.std(all_features, axis=0) + 1e-8

        for sub_id, data in all_subjects_data.items():
            self.subject_stats[sub_id] = {
                'mean': data['features'],
                'std':  global_std
            }

        X_normalized, y_normalized = [], []
        for sub_id, data in all_subjects_data.items():
            subject_mean = self.subject_stats[sub_id]['mean']
            subject_std  = self.subject_stats[sub_id]['std']
            normalized   = (data['features'] - subject_mean) / subject_std
            X_normalized.append(normalized)
            y_normalized.append([data['valence'], data['arousal']])

        X = np.array(X_normalized)
        y = np.array(y_normalized)

        self.model = MultiOutputRegressor(
            RandomForestRegressor(n_estimators=100, random_state=42, max_depth=5)
        )
        self.model.fit(X, y)
        self.avg_baseline = np.mean(all_features, axis=0)
        self.global_std   = global_std
        return len(X)

    def predict_emotion(self, ecg_data, eda_data, subject_id=None):
        if self.model is None:
            raise ValueError("Model not trained yet.")
        features      = self.extract_features(ecg_data, eda_data)
        feature_array = np.array(list(features.values()))

        if subject_id and subject_id in self.subject_stats:
            subject_mean = self.subject_stats[subject_id]['mean']
            subject_std  = self.subject_stats[subject_id]['std']
        else:
            subject_mean = self.avg_baseline
            subject_std  = self.global_std

        feature_normalized = (feature_array - subject_mean) / subject_std
        prediction = self.model.predict(feature_normalized.reshape(1, -1))
        valence = float(np.clip(prediction[0][0], -1, 1))
        arousal = float(np.clip(prediction[0][1], -1, 1))
        return valence, arousal, features


class MusicRecommender:
    def __init__(self, database_path: str):
        self.database = pd.read_csv(database_path)
        self._standardize_columns()
        self._normalize_va_range()

    def _standardize_columns(self):
        rename_map = {
            'valence_tags':   'valence',
            'arousal_tags':   'arousal',
            'dominance_tags': 'dominance'
        }
        self.database = self.database.rename(columns=rename_map)
        missing = [c for c in ['valence', 'arousal'] if c not in self.database.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

    def _normalize_va_range(self):
        # MuSe stores valence/arousal in [0, 1] — remap to [-1, 1]
        if self.database['valence'].min() >= 0 and self.database['valence'].max() <= 1:
            self.database['valence'] = self.database['valence'] * 2 - 1
        if self.database['arousal'].min() >= 0 and self.database['arousal'].max() <= 1:
            self.database['arousal'] = self.database['arousal'] * 2 - 1

    def get_genres(self):
        if 'genre' in self.database.columns:
            return sorted(self.database['genre'].dropna().unique().tolist())
        return []

    def get_emotion_label(self, valence, arousal):
        if valence >= 0 and arousal >= 0:
            return 'Q1', 'Happy / Excited'
        elif valence < 0 and arousal >= 0:
            return 'Q2', 'Angry / Tense'
        elif valence < 0 and arousal < 0:
            return 'Q3', 'Sad / Depressed'
        else:
            return 'Q4', 'Calm / Relaxed'

    def search_by_emotion(self, target_valence, target_arousal, genre=None,
                          num_songs=5, exclude_indices=None):
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

        candidates['distance'] = np.sqrt(
            (candidates['valence'] - target_valence) ** 2 +
            (candidates['arousal'] - target_arousal) ** 2
        )
        return candidates.nsmallest(num_songs, 'distance')

    def recommend_playlist(self, current_valence, current_arousal,
                           target_valence, target_arousal,
                           genre=None, playlist_length=5, gradual=True):
        current_quad, current_desc = self.get_emotion_label(current_valence, current_arousal)
        target_quad,  target_desc  = self.get_emotion_label(target_valence,  target_arousal)

        playlist      = []
        used_indices  = set()

        if gradual and playlist_length >= 3:
            # Start from just past the current emotion, end at the target
            valence_steps = np.linspace(current_valence, target_valence, playlist_length + 1)[1:]
            arousal_steps = np.linspace(current_arousal, target_arousal, playlist_length + 1)[1:]

            for v, a in zip(valence_steps, arousal_steps):
                # Large candidate pool so deduplication doesn't leave us with bad matches
                candidates = self.search_by_emotion(
                    v, a, genre=genre, num_songs=50, exclude_indices=used_indices
                )
                if candidates.empty:
                    continue
                best = candidates.iloc[0]
                playlist.append(best)
                used_indices.add(best.name)
        else:
            songs = self.search_by_emotion(
                target_valence, target_arousal, genre=genre,
                num_songs=playlist_length * 3
            )
            for _, song in songs.iterrows():
                if song.name not in used_indices:
                    playlist.append(song)
                    used_indices.add(song.name)
                if len(playlist) >= playlist_length:
                    break

        return {
            'current_emotion': {
                'valence': current_valence, 'arousal': current_arousal,
                'quadrant': current_quad,   'description': current_desc
            },
            'target_emotion': {
                'valence': target_valence, 'arousal': target_arousal,
                'quadrant': target_quad,   'description': target_desc
            },
            'genre': genre,
            'transition_type': 'gradual' if gradual else 'direct',
            'playlist': playlist[:playlist_length]
        }


class EmotionMusicSystem:
    def __init__(self, music_database_path: str):
        self.predictor  = EmotionPredictor()
        self.recommender = MusicRecommender(music_database_path)
        self.trained     = False

    def train_model(self, physio_dir, annot_dir, progress_cb=None):
        n = self.predictor.train_with_subject_normalization(
            physio_dir, annot_dir, progress_cb=progress_cb
        )
        self.trained = True
        return n

    def process_and_recommend(self, ecg_data, eda_data, target_valence, target_arousal,
                               genre=None, subject_id=None):
        if not self.trained:
            raise ValueError("Model not trained yet.")
        valence, arousal, features = self.predictor.predict_emotion(ecg_data, eda_data, subject_id)
        result = self.recommender.recommend_playlist(
            valence, arousal, target_valence, target_arousal,
            genre=genre, playlist_length=5, gradual=True
        )
        result['extracted_features'] = features
        return result


def generate_dummy_signals(n=10000):
    t   = np.linspace(0, 10, n)
    ecg = np.sin(2 * np.pi * 1.2 * t) + np.random.normal(0, 0.1, n)
    eda = 2 + 0.5 * np.sin(2 * np.pi * 0.05 * t) + np.random.normal(0, 0.05, n)
    return ecg, eda
