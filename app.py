import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import io
import os

st.set_page_config(
    page_title="EmotionBeats",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .main { background: #0f0f13; }
    .block-container { padding: 2rem 2rem 4rem; max-width: 1200px; }

    .hero {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        border-radius: 16px;
        padding: 2.5rem 3rem;
        margin-bottom: 2rem;
        border: 1px solid rgba(100,130,255,0.15);
    }
    .hero h1 { font-size: 2.4rem; font-weight: 700; color: #ffffff; margin: 0 0 .5rem; }
    .hero p  { font-size: 1.05rem; color: #9aa3b0; margin: 0; }
    .accent  { color: #6c8fff; }

    .card {
        background: #16161f;
        border: 1px solid #2a2a38;
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1rem;
    }
    .card-title { font-size: 1rem; font-weight: 600; color: #c8cfe0; margin-bottom: 1rem; letter-spacing: .03em; text-transform: uppercase; }

    .emotion-badge {
        display: inline-block;
        padding: .35rem .9rem;
        border-radius: 20px;
        font-size: .85rem;
        font-weight: 600;
        letter-spacing: .04em;
    }
    .q1 { background: #1a3a1a; color: #5adb5a; border: 1px solid #2d6b2d; }
    .q2 { background: #3a1a1a; color: #db5a5a; border: 1px solid #6b2d2d; }
    .q3 { background: #1a1a3a; color: #5a7adb; border: 1px solid #2d3a6b; }
    .q4 { background: #2a2a1a; color: #c8db5a; border: 1px solid #5a5a2d; }

    .song-card {
        background: #1c1c28;
        border: 1px solid #2e2e40;
        border-radius: 10px;
        padding: 1rem 1.2rem;
        margin-bottom: .6rem;
        display: flex;
        align-items: center;
        gap: 1rem;
    }
    .song-num { font-size: 1.3rem; font-weight: 700; color: #6c8fff; min-width: 28px; }
    .song-info { flex: 1; }
    .song-title  { font-weight: 600; color: #e8ecf5; font-size: .97rem; }
    .song-artist { color: #7a859a; font-size: .85rem; margin-top: .15rem; }
    .song-meta   { font-size: .78rem; color: #555d6e; margin-top: .3rem; }
    .match-pill  {
        background: #1a2540;
        color: #6c8fff;
        padding: .2rem .6rem;
        border-radius: 10px;
        font-size: .75rem;
        font-weight: 600;
        white-space: nowrap;
    }

    .feat-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: .5rem; }
    .feat-item { background: #1c1c28; border-radius: 8px; padding: .6rem .8rem; }
    .feat-name  { font-size: .72rem; color: #606880; text-transform: uppercase; letter-spacing: .05em; }
    .feat-value { font-size: 1rem; font-weight: 600; color: #c8cfe0; margin-top: .15rem; }

    .step-bar { display: flex; align-items: center; gap: .5rem; margin-bottom: 1.5rem; }
    .step { display: flex; align-items: center; gap: .4rem; font-size: .85rem; color: #606880; }
    .step.active { color: #6c8fff; font-weight: 600; }
    .step.done   { color: #3d8f4f; }
    .step-dot { width: 8px; height: 8px; border-radius: 50%; background: currentColor; }
    .step-line { flex: 1; height: 1px; background: #2a2a38; }

    .status-ok   { color: #5adb5a; font-size: .85rem; }
    .status-warn { color: #f0a030; font-size: .85rem; }
    .status-err  { color: #db5a5a; font-size: .85rem; }

    section[data-testid="stSidebar"] { background: #0d0d16 !important; border-right: 1px solid #1e1e2e; }
    section[data-testid="stSidebar"] .block-container { padding: 1rem; }

    #MainMenu, footer { visibility: hidden; }
    header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

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
}.items():
    if k not in st.session_state:
        st.session_state[k] = v

QUAD_CLASS = {"Q1": "q1", "Q2": "q2", "Q3": "q3", "Q4": "q4"}
QUAD_EMOJI = {"Q1": "😄", "Q2": "😠", "Q3": "😢", "Q4": "😌"}

def emotion_badge(desc, quad):
    cls = QUAD_CLASS.get(quad, "q4")
    emoji = QUAD_EMOJI.get(quad, "🎵")
    return f'<span class="emotion-badge {cls}">{emoji} {desc}</span>'

def _get_song_field(song, candidates, fallback="Unknown"):
    for col in candidates:
        val = song.get(col, None)
        if val is not None and str(val).strip() and str(val).strip().lower() != "nan":
            return str(val).strip()
    return fallback

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
        vv = [current_v] + [s.get('valence', 0) if isinstance(s, dict) else s['valence'] for s in playlist] + [target_v]
        aa = [current_a] + [s.get('arousal', 0) if isinstance(s, dict) else s['arousal'] for s in playlist] + [target_a]
        for i in range(len(vv) - 1):
            ax.plot([vv[i], vv[i+1]], [aa[i], aa[i+1]], color="#4a5a8f", lw=1, alpha=.5, zorder=1)

    ax.scatter([current_v], [current_a], s=120, color="#6c8fff", zorder=5, label="Current", edgecolors="#fff", lw=1)
    ax.scatter([target_v], [target_a], s=120, color="#5adb5a", zorder=5, label="Target",  edgecolors="#fff", lw=1, marker="*")

    for side in ax.spines.values():
        side.set_color("#2a2a38")
    ax.tick_params(colors="#505060", labelsize=7)
    ax.set_xlim(-1, 1); ax.set_ylim(-1, 1)
    ax.set_xlabel("Valence →", color="#505060", fontsize=8)
    ax.set_ylabel("Arousal →", color="#505060", fontsize=8)
    ax.legend(fontsize=7, labelcolor="#9aa3b0", facecolor="#1c1c28", edgecolor="#2a2a38")
    plt.tight_layout(pad=.5)
    return fig

def render_playlist(result):
    playlist = result.get("playlist", [])
    if not playlist:
        st.warning("No songs found. Try a different genre or adjust your target emotion.")
        return

    st.markdown('<div class="card-title">🎵 Your Playlist</div>', unsafe_allow_html=True)
    for i, song in enumerate(playlist, 1):
        title  = _get_song_field(song, ['title', 'song_title', 'track', 'name'])
        artist = _get_song_field(song, ['artist', 'artist_name', 'performer'], fallback='Unknown Artist')
        genre  = _get_song_field(song, ['genre', 'genre_tags'], fallback='—')
        v      = float(song.get('valence', 0))
        a      = float(song.get('arousal', 0))
        dist   = float(song.get('distance', 0))

        st.markdown(f"""
        <div class="song-card">
            <div class="song-num">{i}</div>
            <div class="song-info">
                <div class="song-title">{title}</div>
                <div class="song-artist">{artist}</div>
                <div class="song-meta">Genre: {genre} &nbsp;·&nbsp; V={v:.2f}, A={a:.2f}</div>
            </div>
            <div class="match-pill">dist {dist:.3f}</div>
        </div>
        """, unsafe_allow_html=True)

def render_features(features: dict):
    items_html = ""
    for k, v in features.items():
        items_html += f'<div class="feat-item"><div class="feat-name">{k}</div><div class="feat-value">{v:.3f}</div></div>'
    st.markdown(f'<div class="feat-grid">{items_html}</div>', unsafe_allow_html=True)

def render_emotion_summary(result):
    ce = result["current_emotion"]
    te = result["target_emotion"]
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Detected Emotion**")
        st.markdown(emotion_badge(ce["description"], ce["quadrant"]), unsafe_allow_html=True)
        st.markdown(f"<small style='color:#606880'>V={ce['valence']:.2f} · A={ce['arousal']:.2f}</small>", unsafe_allow_html=True)
    with c2:
        st.markdown("**Target Emotion**")
        st.markdown(emotion_badge(te["description"], te["quadrant"]), unsafe_allow_html=True)
        st.markdown(f"<small style='color:#606880'>V={te['valence']:.2f} · A={te['arousal']:.2f}</small>", unsafe_allow_html=True)

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Setup")

    st.markdown("**Music Database (MuSe CSV)**")
    music_db_file = st.file_uploader("Upload muse_dataset.csv", type=["csv"],
                                      label_visibility="collapsed", key="music_db_upload")

    st.divider()
    st.markdown("**CASE Dataset** *(for training)*")
    st.caption("Upload all `sub_N.csv` files from both folders.")
    annot_files  = st.file_uploader("Annotated CSVs", type=["csv"],
                                     accept_multiple_files=True,
                                     key="annot_upload")
    physio_files = st.file_uploader("Physiological CSVs", type=["csv"],
                                     accept_multiple_files=True,
                                     key="physio_upload")

    if st.button("🚀 Initialize & Train", use_container_width=True, type="primary"):
        if not music_db_file:
            st.error("Upload muse_dataset.csv first.")
        else:
            with st.spinner("Loading music database…"):
                try:
                    import tempfile
                    tmp_dir = tempfile.mkdtemp()
                    music_tmp = os.path.join(tmp_dir, "muse_dataset.csv")
                    with open(music_tmp, "wb") as f:
                        f.write(music_db_file.getvalue())
                    system = pl.EmotionMusicSystem(music_tmp)
                    st.session_state.system = system
                except Exception as e:
                    st.error(f"Failed to load music DB: {e}")
                    st.stop()

            if physio_files and annot_files:
                prog = st.progress(0, text="Saving uploaded files…")
                try:
                    physio_tmp = os.path.join(tmp_dir, "Physiological")
                    annot_tmp  = os.path.join(tmp_dir, "Annotated")
                    os.makedirs(physio_tmp, exist_ok=True)
                    os.makedirs(annot_tmp,  exist_ok=True)

                    for f in physio_files:
                        with open(os.path.join(physio_tmp, f.name), "wb") as out:
                            out.write(f.getvalue())
                    for f in annot_files:
                        with open(os.path.join(annot_tmp, f.name), "wb") as out:
                            out.write(f.getvalue())

                    n = st.session_state.system.train_model(
                        physio_tmp, annot_tmp,
                        progress_cb=lambda p: prog.progress(p, text=f"Training… {int(p*100)}%")
                    )
                    prog.empty()
                    st.session_state.trained = True
                    st.success(f"✓ Trained on {n} subjects")
                except Exception as e:
                    prog.empty()
                    st.error(f"Training error: {e}")
            else:
                st.session_state.trained = False
                st.info("Music DB loaded. Upload CASE files to enable full pipeline.")

    st.divider()

    if st.button("🔄 Reset", use_container_width=True):
        for k in ["system", "trained", "result", "demo_result"]:
            st.session_state[k] = None if k in ["system", "result", "demo_result"] else False
        st.rerun()

    st.divider()

    db_ok      = st.session_state.system is not None
    trained_ok = st.session_state.trained
    st.markdown(f'<div class="{"status-ok" if db_ok else "status-warn"}">{"✓" if db_ok else "○"} Music database</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="{"status-ok" if trained_ok else "status-warn"}">{"✓" if trained_ok else "○"} Emotion model</div>', unsafe_allow_html=True)

    st.divider()
    st.markdown("**About**")
    st.caption("ECG + EDA → Emotion (arousal-valence) → Music recommendations via MuSe dataset.")
    st.caption("Uses subject-specific normalization with Random Forest Regressor (CASE dataset).")

# ── HERO ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <h1>🎵 Emotion<span class="accent">Beats</span></h1>
  <p>Physiological signal analysis &rarr; emotion prediction &rarr; personalized music recommendations</p>
</div>
""", unsafe_allow_html=True)

# ── MODE TABS ─────────────────────────────────────────────────────────────────
tab_upload, tab_demo = st.tabs(["📁 Upload Signals", "🎲 Demo Mode"])

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — UPLOAD
# ═══════════════════════════════════════════════════════════════════════════════
with tab_upload:
    if not st.session_state.system:
        st.info("👈 Configure and initialize the system in the sidebar first.")
    else:
        st.markdown("### Upload Physiological Data")
        st.caption("CSV with `ecg` and `gsr` columns at 1000 Hz sampling rate.")

        col_up, col_cfg = st.columns([1, 1], gap="large")

        with col_up:
            uploaded = st.file_uploader("Upload signal CSV", type=["csv"], label_visibility="collapsed")

            subject_id = st.number_input("Subject ID *(optional — used for subject-specific normalization)*",
                                          min_value=1, max_value=30, value=1, step=1,
                                          help="If this subject was in the training set, their baseline is used.")

            if uploaded:
                try:
                    df = pd.read_csv(uploaded)
                    st.success(f"✓ Loaded {len(df):,} samples — columns: {', '.join(df.columns)}")
                    if 'ecg' not in df.columns or 'gsr' not in df.columns:
                        st.error("CSV must have `ecg` and `gsr` columns.")
                        df = None
                except Exception as e:
                    st.error(f"Error reading file: {e}")
                    df = None
            else:
                df = None

        with col_cfg:
            st.markdown("**Target Emotion**")
            genres = st.session_state.system.recommender.get_genres() if st.session_state.system else []
            genre_opts = ["(any)"] + genres
            genre_sel = st.selectbox("Genre", genre_opts)
            genre = None if genre_sel == "(any)" else genre_sel

            t_col1, t_col2 = st.columns(2)
            with t_col1:
                target_v = st.slider("Target Valence", -1.0, 1.0, 0.5, 0.05,
                                      help="−1 = very negative, +1 = very positive")
            with t_col2:
                target_a = st.slider("Target Arousal", -1.0, 1.0, -0.5, 0.05,
                                      help="−1 = very calm, +1 = very energetic")

            tq, td = st.session_state.system.recommender.get_emotion_label(target_v, target_a)
            st.markdown(f"Target: {emotion_badge(td, tq)}", unsafe_allow_html=True)

            st.markdown("**Override Detected Emotion**")
            upload_override = st.toggle("Set current emotion manually", value=False, key="upload_override",
                                         help="By default the model predicts your current emotion from the signal. Enable this to set it yourself.")
            if upload_override:
                uo_col1, uo_col2 = st.columns(2)
                with uo_col1:
                    override_v = st.slider("Current Valence", -1.0, 1.0, 0.0, 0.05, key="uov")
                with uo_col2:
                    override_a = st.slider("Current Arousal", -1.0, 1.0, 0.0, 0.05, key="uoa")
                oq, od = st.session_state.system.recommender.get_emotion_label(override_v, override_a)
                st.markdown(f"From: {emotion_badge(od, oq)}", unsafe_allow_html=True)
            else:
                override_v, override_a = None, None

            playlist_len = st.slider("Playlist length", 3, 10, 5)
            gradual      = st.checkbox("Gradual transition", value=True,
                                        help="Interpolate through intermediate emotions")

        if df is not None:
            if st.button("🔮 Predict & Recommend", type="primary", use_container_width=True):
                if not st.session_state.trained and not upload_override:
                    st.warning("Model not trained — either upload CASE data in the sidebar or enable the emotion override.")
                    st.stop()
                with st.spinner("Extracting features & predicting emotion…"):
                    try:
                        ecg = df["ecg"].values
                        gsr = df["gsr"].values
                        if upload_override:
                            curr_v, curr_a = override_v, override_a
                            feats = {}
                        else:
                            curr_v, curr_a, feats = st.session_state.system.predictor.predict_emotion(
                                ecg, gsr, int(subject_id)
                            )
                        result2 = st.session_state.system.recommender.recommend_playlist(
                            curr_v, curr_a,
                            target_v, target_a, genre=genre,
                            playlist_length=playlist_len, gradual=gradual
                        )
                        result2["extracted_features"] = feats
                        st.session_state.result = result2
                    except Exception as e:
                        st.error(f"Error: {e}")

        if st.session_state.result:
            result = st.session_state.result
            st.divider()

            r_col1, r_col2 = st.columns([1.1, 1], gap="large")

            with r_col1:
                st.markdown('<div class="card-title">Emotion Summary</div>', unsafe_allow_html=True)
                render_emotion_summary(result)
                st.markdown("---")
                fig = va_scatter(
                    result["current_emotion"]["valence"], result["current_emotion"]["arousal"],
                    result["target_emotion"]["valence"],  result["target_emotion"]["arousal"],
                    result.get("playlist", [])
                )
                st.pyplot(fig, use_container_width=False)

                if result.get("extracted_features"):
                    with st.expander("Extracted Features"):
                        render_features(result["extracted_features"])

            with r_col2:
                render_playlist(result)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — DEMO
# ═══════════════════════════════════════════════════════════════════════════════
with tab_demo:
    st.markdown("### Demo Mode")
    st.caption("Set your current and target emotions, pick a genre, and get a playlist.")

    if not st.session_state.system:
        st.info("👈 At minimum, provide the **Music Database** path in the sidebar and click Initialize.")
    else:
        d_col1, d_col2 = st.columns([1, 1], gap="large")

        with d_col1:
            st.markdown("**Current Emotion**")
            dc_col1, dc_col2 = st.columns(2)
            with dc_col1:
                demo_curr_v = st.slider("Current Valence", -1.0, 1.0, -0.4, 0.05, key="dcv")
            with dc_col2:
                demo_curr_a = st.slider("Current Arousal", -1.0, 1.0, 0.3, 0.05, key="dca")
            cq, cd = st.session_state.system.recommender.get_emotion_label(demo_curr_v, demo_curr_a)
            st.markdown(emotion_badge(cd, cq), unsafe_allow_html=True)

        with d_col2:
            st.markdown("**Target Emotion**")
            genres_d = st.session_state.system.recommender.get_genres() if st.session_state.system else []
            genre_opts_d = ["(any)"] + genres_d
            genre_d = st.selectbox("Genre", genre_opts_d, key="demo_genre")
            genre_d = None if genre_d == "(any)" else genre_d

            tv_col1, tv_col2 = st.columns(2)
            with tv_col1:
                demo_tgt_v = st.slider("Target Valence", -1.0, 1.0, 0.6, 0.05, key="dtv")
            with tv_col2:
                demo_tgt_a = st.slider("Target Arousal", -1.0, 1.0, -0.5, 0.05, key="dta")
            tq, td = st.session_state.system.recommender.get_emotion_label(demo_tgt_v, demo_tgt_a)
            st.markdown(emotion_badge(td, tq), unsafe_allow_html=True)

            demo_len = st.slider("Playlist length", 3, 10, 5, key="demo_len")

        if st.button("🎲 Run Demo", type="primary", use_container_width=True):
            with st.spinner("Building playlist…"):
                try:
                    demo_result = st.session_state.system.recommender.recommend_playlist(
                        demo_curr_v, demo_curr_a,
                        demo_tgt_v, demo_tgt_a,
                        genre=genre_d,
                        playlist_length=demo_len,
                        gradual=True
                    )
                    demo_result["extracted_features"] = {}
                    st.session_state.demo_result = demo_result
                except Exception as e:
                    st.error(f"Demo error: {e}")

        if st.session_state.demo_result:
            dr = st.session_state.demo_result
            st.divider()

            dr_col1, dr_col2 = st.columns([1.1, 1], gap="large")

            with dr_col1:
                st.markdown('<div class="card-title">Emotion Summary</div>', unsafe_allow_html=True)
                render_emotion_summary(dr)
                st.markdown("---")
                fig3 = va_scatter(
                    dr["current_emotion"]["valence"], dr["current_emotion"]["arousal"],
                    dr["target_emotion"]["valence"],  dr["target_emotion"]["arousal"],
                    dr.get("playlist", [])
                )
                st.pyplot(fig3, use_container_width=False)

            with dr_col2:
                render_playlist(dr)
