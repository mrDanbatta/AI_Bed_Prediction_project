import pandas as pd
import numpy as np
import psycopg2
from dotenv import load_dotenv
import os

load_dotenv(override=True)
DATABASE_URL = os.getenv("DATABASE_URL")

def get_database_connection():
    conn = psycopg2.connect(DATABASE_URL)
    return conn

def fetch_data(query):
    conn = get_database_connection()
    try:
        df = pd.read_sql_query(query, conn)
    finally:
        conn.close()
    return df

def get_hospital_names():
    if os.path.exists('data/hospital_names.csv'):
        df = pd.read_csv('data/hospital_names.csv')
        return df['hospital'].tolist()
    else:
        query = "SELECT DISTINCT hospital FROM bed_inventory;"
        df = fetch_data(query)
        df = df['hospital']
        df.to_csv('data/hospital_names.csv', index=False)
        return df.tolist()
    
def get_wards(hospital_name):
    if os.path.exists('data/wards.csv'):
        df = pd.read_csv('data/wards.csv')
        return df['ward'].tolist()
    else:
        query = f"SELECT DISTINCT ward FROM bed_inventory WHERE hospital = '{hospital_name}';"
        df = fetch_data(query)
        df = df['ward']
        df.to_csv('data/wards.csv', index=False)
        return df.tolist()

def get_ward_data(hospital_name, ward_name):
    query = f"""
    SELECT * FROM bed_inventory 
    WHERE hospital = '{hospital_name}' AND ward = '{ward_name}';
    """
    df = fetch_data(query)
    df.to_csv(f'data/{hospital_name}_{ward_name}_data.csv', index=False)
    return df

def check_data(hospital_name, ward_name):
    
    file_path = f'data/{hospital_name}_{ward_name}_data.csv'

    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
    else:
        df = get_ward_data(hospital_name, ward_name)
    return df