# Importing necessary libraries
import threading
import mysql.connector

# Creating a thread-local storage for database connections
local_data = threading.local()


# Function to get database connection
def get_db_connection():
    # Check if connection already exists for current thread
    if not hasattr(local_data, 'connection'):
        # Create new connection if not exists
        local_data.connection = mysql.connector.connect(
            host="localhost",
            user="root",
            password="root",
            database="github_issues_db"
        )
    return local_data.connection


# Function to close database connection
def close_db_connection():
    # Check if connection exists for current thread
    if hasattr(local_data, 'connection'):
        # Close connection if exists
        local_data.connection.close()
        del local_data.connection


# Function to execute database query
def execute_query(query, values=None):
    # Get database connection for current thread
    connection = get_db_connection()
    cursor = connection.cursor()
    try:
        # Execute query
        if values:
            cursor.execute(query, values)
        else:
            cursor.execute(query)
        # Commit changes
        connection.commit()
    except:
        # Rollback changes if any error occurs
        connection.rollback()
        raise
    finally:
        # Close cursor
        cursor.close()


# Function to execute database select query
def execute_select_query(query, values=None):
    # Get database connection for current thread
    connection = get_db_connection()
    cursor = connection.cursor()
    try:
        # Execute query
        if values:
            cursor.execute(query, values)
        else:
            cursor.execute(query)
        # Fetch results
        result = cursor.fetchall()
        return result
    except:
        # Rollback changes if any error occurs
        connection.rollback()
        raise
    finally:
        # Close cursor
        cursor.close()
