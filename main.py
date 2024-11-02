import streamlit as st
from authentication_page import show_authentication_page
from reg import show_data_analysis_page
import time

# Define the expiration duration in seconds
SESSION_EXPIRATION_DURATION = 1800  # 30 minutes

# Check if the session has expired
def is_session_expired():
    current_time = time.time()
    last_login_time = st.session_state.get("last_login_time")
    if last_login_time is None:
        return False
    elapsed_time = current_time - last_login_time
    if elapsed_time > SESSION_EXPIRATION_DURATION:
        return True
    return False

# Update the last login time
def update_last_login_time():
    st.session_state["last_login_time"] = time.time()

# Sign out function
def sign_out():
    st.session_state["user_authenticated"] = False
    st.session_state["user_id"] = None
    st.session_state["last_login_time"] = None
    st.rerun()

# Ensure the session state is properly initialized
def initialize_session_state():
    if "user_authenticated" not in st.session_state:
        st.session_state["user_authenticated"] = False
    if "user_id" not in st.session_state:
        st.session_state["user_id"] = None
    if "last_login_time" not in st.session_state:
        st.session_state["last_login_time"] = None

# Main logic to determine which page to show
def main():
    initialize_session_state()

    # Add custom CSS for the header button
    st.markdown(
        """
        <style>
        .header-button {
            position: fixed;
            top: 0;
            right: 0;
            margin: 10px;
            z-index: 100;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Add sign-out button in the header
    if st.session_state["user_authenticated"]:
        st.markdown('<div class="header-button">', unsafe_allow_html=True)
        if st.button("Sign Out"):
            sign_out()
        st.markdown('</div>', unsafe_allow_html=True)

    if is_session_expired():
        st.warning("Session expired. Please sign in again.")
        sign_out()  # Reset the session state
        show_authentication_page()
    else:
        if st.session_state["user_authenticated"]:
            update_last_login_time()  # Reset the timer on user interaction
            show_data_analysis_page()
        else:
            show_authentication_page()

# Call the main function
if __name__ == "__main__":
    main()
