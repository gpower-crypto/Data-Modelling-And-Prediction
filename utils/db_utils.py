import sqlite3
from sqlite3 import Error
import hashlib


def create_connection(db_file):
    conn = None
    try:
        conn = sqlite3.connect(db_file)
        return conn
    except Error as e:
        print(e)
    return conn

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

def fetch_user_datasets(conn, user_id):
    sql_fetch_datasets = """
        SELECT dataset_name, dataset_path FROM datasets WHERE user_id = ?
    """
    try:
        cur = conn.cursor()
        cur.execute(sql_fetch_datasets, (user_id,))
        rows = cur.fetchall()
        datasets = [{"dataset_name": row[0], "dataset_path": row[1]} for row in rows]
        return datasets
    except Error as e:
        print(e)
        return None

def insert_dataset(conn, user_id, dataset_name, dataset_path):
    sql_check_existing_dataset = """
        SELECT dataset_name FROM datasets WHERE user_id = ? AND dataset_name = ?
    """
    sql_insert_dataset = """
        INSERT INTO datasets (user_id, dataset_name, dataset_path)
        VALUES (?, ?, ?)
    """
    try:
        cur = conn.cursor()
        cur.execute(sql_check_existing_dataset, (user_id, dataset_name))
        existing_dataset = cur.fetchone()
        if existing_dataset:
            print(f"Dataset '{dataset_name}' already exists for this user. Please choose a different name.")
            return None
        else:
            cur.execute(sql_insert_dataset, (user_id, dataset_name, dataset_path))
            conn.commit()
            return cur.lastrowid
    except Error as e:
        print(e)
        return None