import streamlit as st

st.set_page_config(
    page_title="Login - Heart Failure Predictor",
    page_icon="🔐",
    layout="centered"
)

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

USERNAME = "admin"
PASSWORD = "1234"

if not st.session_state.logged_in:

    st.title("🔐 Login Page")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if username == USERNAME and password == PASSWORD:
            st.session_state.logged_in = True
            st.success("Login successful!")
            st.switch_page("pages/app.py")
        else:
            st.error("Invalid username or password")
