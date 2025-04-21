
import streamlit as st
import pandas as pd
import joblib


model = joblib.load("titanic_model.pkl")


st.set_page_config(page_title="Titanic Survival Predictor", layout="centered")
st.markdown(
    """
    <style>
    body {
        background-color: #0e1117;
        color: white;
    }
    .stApp {
        background-color: #0e1117;
    }
    </style>
    """, unsafe_allow_html=True
)

st.title("🚢 Titanic Survival Prediction")
st.markdown("أدخل بيانات الراكب وشوف هل سينجو أم لا 👇")

id = st.selectbox("PassengerId",[1])
pclass = st.selectbox("درجة التذكرة (Pclass)", [1, 2, 3])
sex = st.selectbox("الجنس", ["male", "female"])
age = st.slider("العمر", 0, 100, 25)
sibsp = st.number_input("عدد الإخوة/الأزواج على السفينة (SibSp)", 0, 10, 0)
parch = st.number_input("عدد الآباء/الأبناء على السفينة (Parch)", 0, 10, 0)
fare = st.slider("سعر التذكرة (Fare)", 0.0, 600.0, 50.0)
embarked = st.selectbox("ميناء الصعود (Embarked)", ["C", "Q", "S"])

input_data = pd.DataFrame({
    'PassengerId' : [id],
'Pclass': [pclass],
    'Age': [age],
    'SibSp': [sibsp],
    'Parch': [parch],
    'Fare': [fare],
    'Sex_male': [1 if sex == 'male' else 0],
    'Embarked_Q': [1 if embarked == 'Q' else 0],
    'Embarked_S': [1 if embarked == 'S' else 0]
})

input_data = input_data[['PassengerId','Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'Sex_male', 'Embarked_Q', 'Embarked_S']]

if st.button("🔍 تنبؤ"):
    prediction = model.predict(input_data)[0]
    result = "🚨 لن ينجو" if prediction == 0 else "✅ سينجو بإذن الله"
    st.subheader("النتيجة:")
    st.success(result)
