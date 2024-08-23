import sqlite3
import random
import string

def create_table():
    conn = sqlite3.connect('mydatabase.db')
    cursor = conn.cursor()

    create_table_query = '''
    CREATE TABLE IF NOT EXISTS User_Authentication (
        username TEXT PRIMARY KEY,
        password TEXT
    )
    '''

    cursor.execute(create_table_query)
    conn.commit()

    cursor.close()
    conn.close()

def insert_user(username, password):
    conn = sqlite3.connect('mydatabase.db')
    cursor = conn.cursor()

    try:
        insert_query = '''
        INSERT INTO User_Authentication (username, password)
        VALUES (?, ?)
        '''
        cursor.execute(insert_query, (username, password))
        conn.commit()
    except sqlite3.IntegrityError:
        cursor.close()
        conn.close()
        return False

    cursor.close()
    conn.close()
    return True

def verify_credentials(username, password):
    conn = sqlite3.connect('mydatabase.db')
    cursor = conn.cursor()

    select_query = '''
    SELECT * FROM User_Authentication WHERE username = ? AND password = ?
    '''
    cursor.execute(select_query, (username, password))
    result = cursor.fetchone()

    cursor.close()
    conn.close()

    return result is not None

def reset_password(email, new_password):
    conn = sqlite3.connect('mydatabase.db')
    cursor = conn.cursor()

    try:
        # Update the user's password based on username (assuming username is email)
        update_query = '''
        UPDATE User_Authentication
        SET password = ?
        WHERE username = ?
        '''
        cursor.execute(update_query, (new_password, email))
        conn.commit()
        return True
    except Exception as e:
        print("Error resetting password:", str(e))
        return False
    finally:
        cursor.close()
        conn.close()

def show_all_users():
    conn = sqlite3.connect('mydatabase.db')
    cursor = conn.cursor()

    select_all_query = '''
    SELECT * FROM User_Authentication
    '''
    cursor.execute(select_all_query)
    results = cursor.fetchall()

    for row in results:
        print(row)

    cursor.close()
    conn.close()

show_all_users()
