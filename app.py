import streamlit as st
import joblib
import numpy as np

saved = joblib.load("autism_screening_model.joblib")
model = saved["model"]
label_encoders = saved["label_encoders"]
feature_columns = saved["feature_columns"]

st.title("Autism Screening Prediction Tool")
st.write("Answer the following questions. This is a screening aid, not a medical diagnosis.")

# --- Build dropdown options directly from the encoders used in training ---
def options_for(col):
    return [c for c in label_encoders[col].classes_ if str(c) != "nan"]

GENDER_OPTIONS = options_for("gender")
ETHNICITY_OPTIONS = options_for("ethnicity")
JUNDICE_OPTIONS = options_for("jundice")
AUSTIM_OPTIONS = options_for("austim")
COUNTRY_OPTIONS = options_for("contry_of_res")
USED_APP_OPTIONS = options_for("used_app_before")
RELATION_OPTIONS = options_for("relation")

st.subheader("AQ-10 Screening Questions")
questions = [
    "I often notice small sounds when others do not",
    "I usually concentrate more on the whole picture, rather than the small details",
    "I find it easy to do more than one thing at once",
    "If there is an interruption, I can switch back to what I was doing very quickly",
    "I find it easy to 'read between the lines' when someone is talking to me",
    "I know how to tell if someone listening to me is getting bored",
    "When I'm reading a story, I find it difficult to work out the characters' intentions",
    "I like to collect information about categories of things",
    "I find it easy to work out what someone is thinking or feeling just by looking at their face",
    "I find it difficult to work out people's intentions"
]

answers = []
for i, q in enumerate(questions, start=1):
    ans = st.radio(f"A{i}. {q}", ["Agree", "Disagree"], key=f"A{i}")
    answers.append(1 if ans == "Agree" else 0)

st.subheader("Background Information")
age = st.number_input("Age", min_value=1, max_value=100, value=25)
gender = st.selectbox("Gender", GENDER_OPTIONS)
ethnicity = st.selectbox("Ethnicity", ETHNICITY_OPTIONS)
jundice = st.selectbox("Born with jaundice?", JUNDICE_OPTIONS)
austim = st.selectbox("Family member with autism?", AUSTIM_OPTIONS)
country = st.selectbox("Country of residence", COUNTRY_OPTIONS)
used_app_before = st.selectbox("Used a screening app before?", USED_APP_OPTIONS)
relation = st.selectbox("Who is completing this form?", RELATION_OPTIONS)

if st.button("Predict"):
    # Build a row dict in the exact same column order the model was trained on
    row = {
        "A1_Score": answers[0], "A2_Score": answers[1], "A3_Score": answers[2],
        "A4_Score": answers[3], "A5_Score": answers[4], "A6_Score": answers[5],
        "A7_Score": answers[6], "A8_Score": answers[7], "A9_Score": answers[8],
        "A10_Score": answers[9],
        "age": age,
        "gender": label_encoders["gender"].transform([gender])[0],
        "ethnicity": label_encoders["ethnicity"].transform([ethnicity])[0],
        "jundice": label_encoders["jundice"].transform([jundice])[0],
        "austim": label_encoders["austim"].transform([austim])[0],
        "contry_of_res": label_encoders["contry_of_res"].transform([country])[0],
        "used_app_before": label_encoders["used_app_before"].transform([used_app_before])[0],
        "relation": label_encoders["relation"].transform([relation])[0],
    }
    features = np.array([[row[col] for col in feature_columns]])

    prediction = model.predict(features)[0]
    proba = model.predict_proba(features)[0][1]

    st.subheader("Result")
    if prediction == 1:
        st.error(f"Higher likelihood of ASD traits detected (confidence: {proba:.0%})")
    else:
        st.success(f"Lower likelihood of ASD traits detected (confidence: {1 - proba:.0%})")

    st.caption("This tool is for screening purposes only and does not replace a professional diagnosis.")