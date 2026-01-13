import psycopg2


def get_db_connection():
    conn = psycopg2.connect(
        host="localhost",
        port=5432,
        database="iotdb",
        user="iotuser",
        password="iotpass"
    )
    return conn
