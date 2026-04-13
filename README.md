# EmotionBeats — Streamlit App

Emotion-based music recommendation system using ECG + EDA physiological signals.

## Setup

```bash
pip install -r requirements.txt
```

## Run

```bash
streamlit run app.py
```

## Usage

### Sidebar
1. Enter the path to your `muse_dataset.csv` (MuSe database)
2. *(Optional)* Enter paths to the CASE dataset `Physiological/` and `Annotated/` directories
3. Click **Initialize & Train**

### Upload Mode
- Upload a CSV with `ecg` and `gsr` columns (sampled at 1000 Hz)
- Set a subject ID (1–30) if the subject was in the training set
- Pick a target emotion and genre, then click **Predict & Recommend**

### Demo Mode
- No real data needed — generates synthetic signals
- Manually set current/target emotions via sliders
- Click **Run Demo** to see the recommender in action

## File Structure

```
app.py           ← Streamlit UI
pipeline.py      ← EmotionPredictor, MusicRecommender, EmotionMusicSystem
requirements.txt
```
