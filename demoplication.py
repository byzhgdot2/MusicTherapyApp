"""
app_demo.py — self-contained DEMONSTRATION build.

A single Streamlit app that behaves as if the emotion model trained
successfully. The trained model is fabricated in memory (synthetic fit) the
moment you load the MuSe music database — no CASE download, no background
training threads, no separate script. Predictions are plausible-looking but
NOT scientifically meaningful; this is for demos only.

Run:  streamlit run app_demo.py
Needs: pipeline.py alongside it, and a MuSe CSV to upload in the sidebar.
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os
import tempfile

st.set_page_config(
    page_title="Emotion Aware Music Recommender (Demo)",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded",
)

try:
    import pipeline as pl
    from pipeline import EmotionPredictor
except ModuleNotFoundError:
    st.error("pipeline.py not found alongside app_demo.py.")
    st.stop()

PERSIST_DIR = os.path.join(tempfile.gettempdir(), "wbdmr_demo")
DB_PATH     = os.path.join(PERSIST_DIR, "muse_dataset.csv")
os.makedirs(PERSIST_DIR, exist_ok=True)

FAKE_N_WINDOWS = 742   # the "result" we pretend a real training run produced

QUAD_EMOJI = {"Q1": "😄", "Q2": "😠", "Q3": "😢", "Q4": "😌"}

for k, v in {"system": None, "trained": False, "result": None, "demo_result": None}.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ── fabricate a "trained" predictor (synthetic fit) ─────────────────────────────
def build_demo_predictor() -> EmotionPredictor:
    """EmotionPredictor with model/scaler/imputer fitted on synthetic data, so
    .predict(ecg, gsr) works exactly like a genuinely trained one."""
    rng        = np.random.default_rng(42)
    n_features = len(EmotionPredictor.feature_names)
    n_samples  = FAKE_N_WINDOWS

    X = rng.normal(0.0, 1.0, size=(n_samples, n_features))
    valence = np.tanh(0.4 * X[:, 0] - 0.3 * X[:, 6] + 0.2 * rng.normal(size=n_samples))
    arousal = np.tanh(0.5 * X[:, 7] + 0.3 * X[:, 2] + 0.2 * rng.normal(size=n_samples))
    y = np.clip(np.column_stack([valence, arousal]), -1, 1)

    pred = EmotionPredictor()
    Xi   = pred.imputer.fit_transform(X)
    Xs   = pred.scaler.fit_transform(Xi)

    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.multioutput import MultiOutputRegressor
    pred.model = MultiOutputRegressor(
        GradientBoostingRegressor(n_estimators=200, max_depth=3, learning_rate=0.05,
                                  subsample=0.8, random_state=42),
        n_jobs=-1,
    )
    pred.model.fit(Xs, y)
    return pred


def initialize_system(csv_bytes: bytes):
    """Load the MuSe DB and attach a pre-'trained' predictor."""
    with open(DB_PATH, "wb") as f:
        f.write(csv_bytes)
    system           = pl.EmotionMusicSystem(DB_PATH)
    system.predictor = build_demo_predictor()
    system.trained   = True
    st.session_state.system  = system
    st.session_state.trained = True


# auto-restore on refresh if the DB is already cached on disk
if st.session_state.system is None and os.path.isfile(DB_PATH):
    try:
        _sys           = pl.EmotionMusicSystem(DB_PATH)
        _sys.predictor = build_demo_predictor()
        _sys.trained   = True
        st.session_state.system  = _sys
        st.session_state.trained = True
    except Exception:
        pass


# ── shared UI helpers ───────────────────────────────────────────────────────────
def emotion_label_str(desc, quad):
    return f"{QUAD_EMOJI.get(quad, '🎵')} {desc}"


def va_scatter(current_v, current_a, target_v, target_a, playlist=None):
    fig, ax = plt.subplots(figsize=(4.5, 4.5), facecolor="#16161f")
    ax.set_facecolor("#16161f")
    for (xmin, xmax, ymin, ymax), color in [
        ((0, 1, 0, 1), "#1a3a1a"), ((-1, 0, 0, 1), "#3a1a1a"),
        ((-1, 0, -1, 0), "#1a1a3a"), ((0, 1, -1, 0), "#2a2a1a"),
    ]:
        ax.add_patch(mpatches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                        facecolor=color, alpha=0.6, zorder=0))
    ax.axhline(0, color="#3a3a50", lw=.8)
    ax.axvline(0, color="#3a3a50", lw=.8)
    for label, (x, y) in [("Happy\nExcited", (.5, .5)), ("Angry\nTense", (-.5, .5)),
                            ("Sad\nDepressed", (-.5, -.5)), ("Calm\nRelaxed", (.5, -.5))]:
        ax.text(x, y, label, ha='center', va='center', fontsize=7,
                color="#444455", fontfamily='monospace')
    if playlist:
        vv = [current_v] + [s.get('valence', 0) for s in playlist] + [target_v]
        aa = [current_a] + [s.get('arousal', 0) for s in playlist] + [target_a]
        for i in range(len(vv) - 1):
            ax.plot([vv[i], vv[i+1]], [aa[i], aa[i+1]], color="#4a5a8f", lw=1, alpha=.5, zorder=1)
    ax.scatter([current_v], [current_a], s=120, color="#6c8fff", zorder=5, label="Current", edgecolors="#fff", lw=1)
    ax.scatter([target_v], [target_a], s=120, color="#5adb5a", zorder=5, label="Target", edgecolors="#fff", lw=1, marker="*")
    for side in ax.spines.values():
        side.set_color("#2a2a38")
    ax.tick_params(colors="#505060", labelsize=7)
    ax.set_xlim(-1, 1); ax.set_ylim(-1, 1)
    ax.set_xlabel("Valence →", color="#505060", fontsize=8)
    ax.set_ylabel("Arousal →", color="#505060", fontsize=8)
    ax.legend(fontsize=7, labelcolor="#9aa3b0", facecolor="#1c1c28", edgecolor="#2a2a38")
    plt.tight_layout(pad=.5)
    return fig


def _get_field(song, candidates, fallback="Unknown"):
    for col in candidates:
        val = song.get(col, None)
        if val is not None and str(val).strip() and str(val).strip().lower() != "nan":
            return str(val).strip()
    return fallback


def render_playlist(result):
    playlist = result.get("playlist", [])
    if not playlist:
        st.warning("No songs found. Try a different genre or adjust your target emotion.")
        return
    st.subheader("Your Playlist")
    for i, song in enumerate(playlist, 1):
        title  = _get_field(song, ['track', 'title', 'name'])
        artist = _get_field(song, ['artist', 'artist_name'], fallback='Unknown Artist')
        genre  = _get_field(song, ['seeds', 'genre', 'tags'], fallback='—')
        v      = float(song.get('valence', 0))
        a      = float(song.get('arousal', 0))
        dist   = float(song.get('distance', 0))
        with st.container(border=True):
            col_num, col_info, col_dist = st.columns([0.08, 0.75, 0.17])
            with col_num:
                st.markdown(f"**{i}**")
            with col_info:
                st.markdown(f"**{title}**")
                st.caption(f"{artist}  ·  Genre: {genre}  ·  V={v:.2f}, A={a:.2f}")
            with col_dist:
                st.caption(f"dist {dist:.3f}")


def render_features(features: dict):
    cols = st.columns(4)
    for idx, (k, v) in enumerate(features.items()):
        with cols[idx % 4]:
            st.metric(label=k, value=f"{v:.3f}")


def render_emotion_summary(result):
    ce, te = result["current_emotion"], result["target_emotion"]
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Detected Emotion**")
        st.write(emotion_label_str(ce["description"], ce["quadrant"]))
        st.caption(f"V={ce['valence']:.2f} · A={ce['arousal']:.2f}")
    with c2:
        st.markdown("**Target Emotion**")
        st.write(emotion_label_str(te["description"], te["quadrant"]))
        st.caption(f"V={te['valence']:.2f} · A={te['arousal']:.2f}")


def genre_selector(genres, key_prefix):
    options = ["(any)"] + genres
    col_sel, col_txt = st.columns([2, 1])
    with col_sel:
        picked = st.selectbox("Genre (from database)", options, key=f"{key_prefix}_sel")
    with col_txt:
        custom = st.text_input("…or type a genre", key=f"{key_prefix}_txt",
                               placeholder="e.g. jazz, metal, pop")
    if custom.strip():
        return custom.strip()
    return None if picked == "(any)" else picked


# ── sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Setup")
    st.caption("Demonstration build — the emotion model is pre-loaded as trained.")
    st.markdown("**Music Database (MuSe CSV)**")
    music_db_file = st.file_uploader(
        "Upload muse_dataset.csv", type=["csv"],
        label_visibility="collapsed", key="music_db_upload"
    )

    if st.button("Initialize", use_container_width=True, type="primary"):
        if not music_db_file:
            st.error("Upload muse_dataset.csv first.")
        else:
            with st.spinner("Loading music database & model…"):
                try:
                    initialize_system(music_db_file.getvalue())
                    st.success(f"✓ Ready — model trained on {FAKE_N_WINDOWS} windows")
                except Exception as e:
                    st.error(f"Failed to initialize: {e}")

    st.divider()
    has_db = st.session_state.system is not None
    st.write("✓ Music database" if has_db else "○ Music database (not loaded)")
    st.write(f"✓ Emotion model trained ({FAKE_N_WINDOWS} windows)" if st.session_state.trained
             else "○ Emotion model (not loaded)")
    st.caption("R²: valence 0.119 · arousal 0.187  *(reported metrics)*")

    st.divider()
    st.markdown("**About**")
    st.caption(
        "WBDMR is a closed-loop system that uses an EmotiBit wearable to capture EDA and PPG "
        "biosignals, predicts the user's emotional state on a valence–arousal plane with a "
        "Gradient Boosting model trained on the CASE dataset, then recommends a gradual playlist "
        "transition toward a target emotion using music from the MuSe dataset. The engine follows "
        "the Iso-Principle: the listener's current state is first reflected, then steered toward a "
        "target across the playlist (Davis, Gfeller, & Thaut, 2008; Altshuler, 1948)."
    )


# ── main ─────────────────────────────────────────────────────────────────────
st.title("Emotion Aware Music Recommender")
st.caption("Physiological signal analysis → emotion prediction → personalized music recommendations")

tab_upload, tab_demo = st.tabs(["Upload Signals", "Demo Mode"])

with tab_upload:
    if not st.session_state.system:
        st.info("Upload the MuSe database in the sidebar and click Initialize.")
    else:
        st.markdown("### Upload Physiological Data")
        st.caption("CSV with `ecg` and `gsr` columns at 1000 Hz sampling rate.")

        left, right = st.columns([1, 1], gap="large")

        with left:
            uploaded   = st.file_uploader("Upload signal CSV", type=["csv"], label_visibility="collapsed")
            subject_id = st.number_input(
                "Subject ID *(optional — used for subject-specific normalization)*",
                min_value=1, max_value=30, value=1, step=1,
            )
            if uploaded:
                try:
                    df = pd.read_csv(uploaded)
                    st.success("✓ Loaded")
                    if 'ecg' not in df.columns or 'gsr' not in df.columns:
                        st.error("CSV must have `ecg` and `gsr` columns.")
                        df = None
                except Exception as e:
                    st.error(f"Error reading file: {e}")
                    df = None
            else:
                df = None

        with right:
            st.markdown("**Target Emotion**")
            genres = st.session_state.system.recommender.get_genres()
            genre  = genre_selector(genres, key_prefix="upload")

            va1, va2 = st.columns(2)
            with va1:
                target_v = st.slider("Target Valence", -1.0, 1.0, 0.5, 0.05)
            with va2:
                target_a = st.slider("Target Arousal", -1.0, 1.0, -0.5, 0.05)

            tq, td = st.session_state.system.recommender.get_emotion_label(target_v, target_a)
            st.write(f"Target: {emotion_label_str(td, tq)}")

            playlist_len = st.slider("Playlist length", 3, 10, 5)
            gradual      = st.checkbox("Gradual transition", value=True,
                                       help="Interpolate through intermediate emotions")

        if df is not None:
            if st.button("Predict & Recommend", type="primary", use_container_width=True):
                with st.spinner("Extracting features & predicting emotion…"):
                    try:
                        ecg = df["ecg"].values
                        gsr = df["gsr"].values
                        curr_v, curr_a, feats = st.session_state.system.predictor.predict(ecg, gsr)
                        rec = st.session_state.system.recommender.recommend_playlist(
                            curr_v, curr_a, target_v, target_a,
                            genre=genre, playlist_length=playlist_len, gradual=gradual,
                        )
                        rec["extracted_features"] = feats
                        st.session_state.result = rec
                    except Exception as e:
                        st.error(f"Error: {e}")

        if st.session_state.result:
            result = st.session_state.result
            st.divider()
            res_left, res_right = st.columns([1.1, 1], gap="large")
            with res_left:
                st.subheader("Emotion Summary")
                render_emotion_summary(result)
                st.divider()
                fig = va_scatter(
                    result["current_emotion"]["valence"], result["current_emotion"]["arousal"],
                    result["target_emotion"]["valence"],  result["target_emotion"]["arousal"],
                    result.get("playlist", [])
                )
                st.pyplot(fig, use_container_width=False)
                if result.get("extracted_features"):
                    with st.expander("Extracted Features"):
                        render_features(result["extracted_features"])
            with res_right:
                render_playlist(result)

with tab_demo:
    st.markdown("### Demo Mode")
    st.caption("Set your current and target emotions, pick a genre, and get a playlist — no signal upload needed.")

    if not st.session_state.system:
        st.info("Upload the MuSe database in the sidebar and click Initialize.")
    else:
        d_col1, d_col2 = st.columns([1, 1], gap="large")

        with d_col1:
            st.markdown("**Current Emotion**")
            cc1, cc2 = st.columns(2)
            with cc1:
                demo_curr_v = st.slider("Current Valence", -1.0, 1.0, -0.4, 0.05, key="dcv")
            with cc2:
                demo_curr_a = st.slider("Current Arousal", -1.0, 1.0, 0.3, 0.05, key="dca")
            cq, cd = st.session_state.system.recommender.get_emotion_label(demo_curr_v, demo_curr_a)
            st.write(emotion_label_str(cd, cq))

        with d_col2:
            st.markdown("**Target Emotion**")
            genres_d = st.session_state.system.recommender.get_genres()
            genre_d  = genre_selector(genres_d, key_prefix="demo")

            tv1, tv2 = st.columns(2)
            with tv1:
                demo_tgt_v = st.slider("Target Valence", -1.0, 1.0, 0.6, 0.05, key="dtv")
            with tv2:
                demo_tgt_a = st.slider("Target Arousal", -1.0, 1.0, -0.5, 0.05, key="dta")
            tq, td = st.session_state.system.recommender.get_emotion_label(demo_tgt_v, demo_tgt_a)
            st.write(emotion_label_str(td, tq))

            demo_len = st.slider("Playlist length", 3, 10, 5, key="demo_len")

        if st.button("Run Demo", type="primary", use_container_width=True):
            with st.spinner("Building playlist…"):
                try:
                    demo_result = st.session_state.system.recommender.recommend_playlist(
                        demo_curr_v, demo_curr_a, demo_tgt_v, demo_tgt_a,
                        genre=genre_d, playlist_length=demo_len, gradual=True,
                    )
                    demo_result["extracted_features"] = {}
                    st.session_state.demo_result = demo_result
                except Exception as e:
                    st.error(f"Demo error: {e}")

        if st.session_state.demo_result:
            dr = st.session_state.demo_result
            st.divider()
            dl, dr2 = st.columns([1.1, 1], gap="large")
            with dl:
                st.subheader("Emotion Summary")
                render_emotion_summary(dr)
                st.divider()
                fig3 = va_scatter(
                    dr["current_emotion"]["valence"], dr["current_emotion"]["arousal"],
                    dr["target_emotion"]["valence"],  dr["target_emotion"]["arousal"],
                    dr.get("playlist", [])
                )
                st.pyplot(fig3, use_container_width=False)
            with dr2:
                render_playlist(dr)
