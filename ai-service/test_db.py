import psycopg2
print("DB test started")

conn = psycopg2.connect(
    host="127.0.0.1",
    port=5432,
    dbname="iotdb",
    user="iotuser",
    password="iotpass",
    connect_timeout=5
)

print("Connected to DB")

cur = conn.cursor()
cur.execute("SELECT COUNT(*) FROM telemetry;")
count = cur.fetchone()[0]

print("Telemetry row count:", count)

cur.close()
conn.close()

