import streamlit as st
import pandas as pd

st.title("Emotion-Based Music Recommender")

uploaded = st.file_uploader("Upload physiological signals (.csv)", type="csv")

if uploaded:
    df = pd.read_csv(uploaded)
    
    with st.spinner("Predicting emotional state..."):
        valence, arousal = predict_emotion(df)  # your existing function
    
    st.metric("Predicted Valence", round(valence, 3))
    st.metric("Predicted Arousal", round(arousal, 3))
    
    st.subheader("Set your target emotion")
    target_v = st.slider("Target Valence", -1.0, 1.0, 0.0)
    target_a = st.slider("Target Arousal", -1.0, 1.0, 0.0)
    
    if st.button("Get Playlist"):
        playlist = recommend(valence, arousal, target_v, target_a)  # your existing function
        st.dataframe(playlist)