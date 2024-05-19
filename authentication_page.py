import streamlit as st
import time
from utils.db_utils import create_connection, create_user_table, create_dataset_table, insert_user, authenticate_user

# Update the last login time
def update_last_login_time():
    st.session_state["last_login_time"] = time.time()

def show_authentication_page():
    # Create or connect to the database
    conn = create_connection("user_data.db")

    # Create tables if they don't exist
    if conn is not None:
        create_user_table(conn)
        create_dataset_table(conn)
    else:
        st.error("Error: Could not connect to the database.")

    # Title and description
    st.title("Authentication Page")

    # Checkbox to toggle between login and signup
    login_or_signup = st.checkbox("New User? Sign Up")

    if login_or_signup:
        # Sign up form
        signup_username = st.text_input("Username", key="signup_username")
        signup_password = st.text_input("Password", type="password", key="signup_password")
        signup_button = st.button("Sign Up", key="signup_button")

        if signup_button:
            if signup_username and signup_password:
                # Insert new user into the database
                user_id = insert_user(conn, signup_username, signup_password)
                if user_id is not None:
                    st.success("Signup successful!")
                    st.session_state["user_authenticated"] = True
                    st.session_state["user_id"] = user_id
                    update_last_login_time()  # Update the last login time
                    st.rerun()
                else:
                    st.error("Error: Username already exists. Please choose a different username.")
            else:
                st.warning("Please enter both username and password.")
    else:
        # Login form
        login_username = st.text_input("Username", key="login_username")
        login_password = st.text_input("Password", type="password", key="login_password")
        login_button = st.button("Login", key="login_button")

        # Check if the login button is clicked
        if login_button:
            if login_username and login_password:
                # Authenticate the user
                user_id = authenticate_user(conn, login_username, login_password)
                if user_id is not None:
                    st.session_state["user_authenticated"] = True
                    st.session_state["user_id"] = user_id
                    update_last_login_time()  # Update the last login time
                    st.success("Login successful!")
                    st.write(f"Welcome, {login_username}!")
                    st.rerun()
                else:
                    st.error("Invalid username or password. Please try again.")
            else:
                st.warning("Please enter both username and password.")
