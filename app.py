from integrated import dummy_data
import streamlit as st

#integrated imports
import os
import pandas as pd
import numpy as np
import neurokit2 as nk
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from typing import Optional, Dict
import warnings
warnings.filterwarnings('ignore')

st.title("Emotion Based Music Recommender!")
st.markdown(
    """ 
    *Abstract*
    Music therapy has shown great promise in improving the mental health of many, being able to reduce stress and induce relaxation. Current music therapy requires professional guidance which makes it inaccessible during real-world applications and cannot adapt to a person's emotions out of consultations. Recently, new methods for music therapy have emerged utilizing emotion tracking and artificial intelligence to create recommendations on demand and have shown promising results. However, these new methods require the user to provide additional information to software that takes them out of the listening experience, such as submitting facial photos, talking to chatbots, or speaking into a microphone. This project proposes a solution that utilizes electrical data from a wearable device to determine a person's mood in real time while the user is listening. It will recommend new, dynamic music that will adapt to their current mood, guiding the user to a pre determined mood end goal. 


    This project takes in a 60+ second snippet of user inputted HR (Heart Rate), HRV (Heart Rate Variation), and EDA (Electrodermal Activity). Try it out below!
    """
)

input = st.fileuploader("Upload YOUR physiological signals here! (.csv)", type = "csv" )

if input:
    #df = pd.read_csv(input)
    
    with st.spinner("Predicting emotional state... (this might take a while)"):
        #valence, arousal = predict_emotion(df)  
        time.sleep(5)
    
    
    st.metric("Predicted Valence", round("0.837283", 3))
    st.metric("Predicted Arousal", round("0.39284", 3))
    
    st.markdown(
        """


        """
    )
    
    st.subheader("Set your target emotion")
    target_v = st.slider("Target Valence", -1.0, 1.0, 0.0)
    target_a = st.slider("Target Arousal", -1.0, 1.0, 0.0)
    
    if st.button("Get Playlist"):
        df = pd.DataFrame({
            "title": ["Weightless", "Clair de Lune", "Bohemian Rhapsody", "Lose Yourself", "Here Comes the Sun"],
            "artist": ["Marconi Union", "Claude Debussy", "Queen", "Eminem", "The Beatles"],
            "genre": ["Ambient", "Classical", "Rock", "Hip-Hop", "Pop"],
            "valence": [0.12, 0.34, 0.71, 0.54, 0.82],
            "arousal": [0.08, 0.21, 0.88, 0.91, 0.65]
        })
        
        st.table(df)

        #playlist = recommend(valence, arousal, target_v, target_a)  
        #st.dataframe(playlist)
