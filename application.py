import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os
import tempfile

st.set_page_config(
    page_title="Emotion Aware Music Recommender",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded",
)

try:
    import pipeline as pl
except ModuleNotFoundError:
    st.error("pipeline.py not found alongside app.py.")
    st.stop()

for k, v in {
    "system": None,
    "trained": False,
    "result": None,
    "demo_result": None,
    "dataset_ready": False,
    "physio_dir": None,
    "annot_dir": None,
}.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ── Google Drive folder IDs ────────────────────────────────────────────────────
GDRIVE_ANNOT_FOLDER_ID  = "1WZfE0gnPkvgIHfsvdY_7PUDQ-r05oGjZ"
GDRIVE_PHYSIO_FOLDER_ID = "1jqz4YcJcCpwLAP6PAfeGzrwNxqomfqis"

QUAD_EMOJI = {"Q1": "😄", "Q2": "😠", "Q3": "😢", "Q4": "😌"}

def emotion_label_str(desc, quad):
    emoji = QUAD_EMOJI.get(quad, "🎵")
    return f"{emoji} {desc}"

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
    ce = result["current_emotion"]
    te = result["target_emotion"]
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


# ── Google Drive download helper ───────────────────────────────────────────────
def download_gdrive_folder(folder_id: str, dest_dir: str, label: str, progress_bar):
    """
    Download all files in a public Google Drive folder using gdown.
    Returns the number of files downloaded.
    """
    try:
        import gdown
    except ImportError:
        st.error("gdown not installed. Add `gdown` to requirements.txt.")
        st.stop()

    os.makedirs(dest_dir, exist_ok=True)

    # List files in folder via gdown
    url = f"https://drive.google.com/drive/folders/{folder_id}"
    try:
        # gdown.download_folder downloads everything into dest_dir
        gdown.download_folder(url, output=dest_dir, quiet=True, use_cookies=False)
    except Exception as e:
        st.error(f"Failed to download {label} from Google Drive: {e}")
        return 0

    files = [f for f in os.listdir(dest_dir)
             if f.endswith((".csv", ".xlsx", ".xls"))]
    return len(files)


# ═══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("### Setup")

    st.markdown("**Music Database (MuSe CSV)**")
    music_db_file = st.file_uploader(
        "Upload muse_dataset.csv", type=["csv"],
        label_visibility="collapsed", key="music_db_upload"
    )

    st.divider()

    st.markdown("**CASE Dataset**")
    if st.session_state.dataset_ready:
        physio_files = [f for f in os.listdir(st.session_state.physio_dir)
                        if f.endswith((".csv", ".xlsx", ".xls"))]
        annot_files  = [f for f in os.listdir(st.session_state.annot_dir)
                        if f.endswith((".csv", ".xlsx", ".xls"))]
        st.success(
            f"✓ Dataset ready  \n"
            f"{len(physio_files)} physio · {len(annot_files)} annotation files"
        )
    else:
        st.caption(
            "The physiological and annotation files will be downloaded automatically "
            "from Google Drive when you click Initialize."
        )

    st.divider()

    if st.button("Initialize", use_container_width=True, type="primary"):
        if not music_db_file:
            st.error("Upload muse_dataset.csv first.")
        else:
            # ── Load music DB ──────────────────────────────────────────────
            with st.spinner("Loading music database…"):
                try:
                    tmpdir = tempfile.mkdtemp()
                    db_path = os.path.join(tmpdir, "muse_dataset.csv")
                    with open(db_path, "wb") as f:
                        f.write(music_db_file.getvalue())
                    system = pl.EmotionMusicSystem(db_path)
                    st.session_state.system = system
                    st.success("✓ Loaded")
                except Exception as e:
                    st.error(f"Failed to load music DB: {e}")
                    st.stop()

            # ── Download CASE dataset from Google Drive ────────────────────
            if not st.session_state.dataset_ready:
                base_tmp   = tempfile.mkdtemp()
                physio_dir = os.path.join(base_tmp, "Physiological")
                annot_dir  = os.path.join(base_tmp, "Annotated")

                st.info("Downloading CASE dataset from Google Drive…")
                prog = st.progress(0, text="Downloading physiological files…")

                n_physio = download_gdrive_folder(
                    GDRIVE_PHYSIO_FOLDER_ID, physio_dir, "Physiological", prog
                )
                prog.progress(0.5, text="Downloading annotation files…")

                n_annot = download_gdrive_folder(
                    GDRIVE_ANNOT_FOLDER_ID, annot_dir, "Annotated", prog
                )
                prog.empty()

                if n_physio == 0 or n_annot == 0:
                    st.error(
                        f"Download incomplete — {n_physio} physio, {n_annot} annotation files found. "
                        "Check that the Google Drive folders are shared publicly."
                    )
                else:
                    st.session_state.physio_dir    = physio_dir
                    st.session_state.annot_dir     = annot_dir
                    st.session_state.dataset_ready = True
                    st.success(f"✓ Downloaded {n_physio} physio, {n_annot} annotation files")

            # ── Train model ────────────────────────────────────────────────
            if st.session_state.dataset_ready:
                prog2 = st.progress(0, text="Training model…")
                try:
                    n = st.session_state.system.train_model(
                        st.session_state.physio_dir,
                        st.session_state.annot_dir,
                        progress_cb=lambda p: prog2.progress(p, text=f"Training… {int(p*100)}%")
                    )
                    prog2.empty()
                    st.session_state.trained = True
                    st.success(f"✓ Trained on {n} windows")
                except Exception as e:
                    prog2.empty()
                    st.error(f"Training error: {e}")

    st.divider()

    has_db     = st.session_state.system is not None
    is_trained = st.session_state.trained
    st.write("✓ Music database" if has_db else "○ Music database (not loaded)")
    st.write("✓ Dataset downloaded" if st.session_state.dataset_ready else "○ Dataset (will download on Initialize)")
    st.write("✓ Emotion model trained" if is_trained else "○ Emotion model (not trained)")

    st.divider()
    st.markdown("**About**")
    st.caption(
        "Music therapy has shown great promise in improving mental health, reducing stress and inducing relaxation. "
        "Current music therapy requires professional guidance, making it inaccessible in real-world settings. "
        "This project uses electrical data from a wearable device to determine a person's mood in real time "
        "and recommends music that guides the user toward a predetermined mood end goal — no extra input required."
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN AREA
# ═══════════════════════════════════════════════════════════════════════════════
st.title("Emotion Aware Music Recommender")
st.caption("Physiological signal analysis → emotion prediction → personalized music recommendations")

tab_upload, tab_demo = st.tabs(["Upload Signals", "Demo Mode"])

# ─── TAB 1: Upload Signals ────────────────────────────────────────────────────
with tab_upload:
    if not st.session_state.system:
        st.info("Configure and initialize the system in the sidebar first.")
    else:
        st.markdown("### Upload Physiological Data")
        st.caption("CSV with `ecg` and `gsr` columns at 1000 Hz sampling rate.")

        left, right = st.columns([1, 1], gap="large")

        with left:
            uploaded = st.file_uploader("Upload signal CSV", type=["csv"],
                                        label_visibility="collapsed")
            subject_id = st.number_input(
                "Subject ID *(optional — used for subject-specific normalization)*",
                min_value=1, max_value=30, value=1, step=1,
                help="If this subject was in the training set, their baseline is used."
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
            genre = genre_selector(genres, key_prefix="upload")

            va1, va2 = st.columns(2)
            with va1:
                target_v = st.slider("Target Valence", -1.0, 1.0, 0.5, 0.05,
                                     help="−1 = very negative, +1 = very positive")
            with va2:
                target_a = st.slider("Target Arousal", -1.0, 1.0, -0.5, 0.05,
                                     help="−1 = very calm, +1 = very energetic")

            tq, td = st.session_state.system.recommender.get_emotion_label(target_v, target_a)
            st.write(f"Target: {emotion_label_str(td, tq)}")

            st.markdown("**Override Detected Emotion**")
            manual_mode = st.toggle(
                "Set current emotion manually", value=False, key="manual_mode",
                help="By default the model predicts your current emotion from the signal. "
                     "Enable this if the model is not trained."
            )
            if manual_mode:
                oc1, oc2 = st.columns(2)
                with oc1:
                    override_v = st.slider("Current Valence", -1.0, 1.0, 0.0, 0.05, key="uov")
                with oc2:
                    override_a = st.slider("Current Arousal", -1.0, 1.0, 0.0, 0.05, key="uoa")
                oq, od = st.session_state.system.recommender.get_emotion_label(override_v, override_a)
                st.write(f"From: {emotion_label_str(od, oq)}")
            else:
                override_v, override_a = None, None

            playlist_len = st.slider("Playlist length", 3, 10, 5)
            gradual      = st.checkbox("Gradual transition", value=True,
                                       help="Interpolate through intermediate emotions")

        if df is not None:
            if st.button("Predict & Recommend", type="primary", use_container_width=True):
                if not st.session_state.trained and not manual_mode:
                    st.warning(
                        "Model not trained — click Initialize in the sidebar, "
                        "or enable the emotion override toggle above."
                    )
                    st.stop()
                with st.spinner("Extracting features & predicting emotion…"):
                    try:
                        ecg = df["ecg"].values
                        gsr = df["gsr"].values
                        if manual_mode:
                            curr_v, curr_a = override_v, override_a
                            feats = {}
                        else:
                            curr_v, curr_a, feats = st.session_state.system.predictor.predict(ecg, gsr)
                        rec = st.session_state.system.recommender.recommend_playlist(
                            curr_v, curr_a,
                            target_v, target_a,
                            genre=genre,
                            playlist_length=playlist_len,
                            gradual=gradual,
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

# ─── TAB 2: Demo Mode ─────────────────────────────────────────────────────────
with tab_demo:
    st.markdown("### Demo Mode")
    st.caption("Set your current and target emotions, pick a genre, and get a playlist — no signal upload needed.")

    if not st.session_state.system:
        st.info("At minimum, provide the Music Database in the sidebar and click Initialize.")
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
            genre_d = genre_selector(genres_d, key_prefix="demo")

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
                        demo_curr_v, demo_curr_a,
                        demo_tgt_v, demo_tgt_a,
                        genre=genre_d,
                        playlist_length=demo_len,
                        gradual=True,
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
