# authentication_page.py

import streamlit as st
import hashlib
import sqlite3
from sqlite3 import Error
import time

# Update the last login time
def update_last_login_time():
    st.session_state["last_login_time"] = time.time()

def show_authentication_page():
    # Function to create an SQLite database connection
    def create_connection(db_file):
        conn = None
        try:
            conn = sqlite3.connect(db_file)
            return conn
        except Error as e:
            print(e)
        return conn

    # Function to create a new user table
    def create_user_table(conn):
        sql_create_users_table = """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL UNIQUE,
                password TEXT NOT NULL
            );
        """
        try:
            c = conn.cursor()
            c.execute(sql_create_users_table)
        except Error as e:
            print(e)

    # Function to create a new dataset table
    def create_dataset_table(conn):
        sql_create_datasets_table = """
            CREATE TABLE IF NOT EXISTS datasets (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                dataset_name TEXT,
                dataset_path TEXT,
                FOREIGN KEY (user_id) REFERENCES users (id)
            );
        """
        try:
            c = conn.cursor()
            c.execute(sql_create_datasets_table)
        except Error as e:
            print(e)

    # Function to insert a new user into the users table
    def insert_user(conn, username, password):
        sql_insert_user = """
            INSERT INTO users (username, password)
            VALUES (?, ?)
        """
        hashed_password = hashlib.sha256(password.encode()).hexdigest()
        try:
            cur = conn.cursor()
            cur.execute(sql_insert_user, (username, hashed_password))
            conn.commit()
            return cur.lastrowid
        except Error as e:
            print(e)

    # Function to authenticate a user
    def authenticate_user(conn, username, password):
        sql_select_user = """
            SELECT id, password FROM users WHERE username = ?
        """
        hashed_password = hashlib.sha256(password.encode()).hexdigest()
        try:
            cur = conn.cursor()
            cur.execute(sql_select_user, (username,))
            user = cur.fetchone()
            if user and user[1] == hashed_password:
                return user[0]  # Return the user id
            else:
                return None
        except Error as e:
            print(e)

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
