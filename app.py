import streamlit as st

st.title("Emotion Based Music Recommender!")
st.markdown(
    """ 
    Music therapy has shown great promise in improving the mental health of many, being able to reduce stress and induce relaxation. Current music therapy requires professional guidance which makes it inaccessible during real-world applications and cannot adapt to a person's emotions out of consultations. Recently, new methods for music therapy have emerged utilizing emotion tracking and artificial intelligence to create recommendations on demand and have shown promising results. However, these new methods require the user to provide additional information to software that takes them out of the listening experience, such as submitting facial photos, talking to chatbots, or speaking into a microphone. This project proposes a solution that utilizes electrical data from a wearable device to determine a person's mood in real time while the user is listening. It will recommend new, dynamic music that will adapt to their current mood, guiding the user to a pre determined mood end goal. 

    """
)

if st.button("Send balloons!"):
    st.balloons()
