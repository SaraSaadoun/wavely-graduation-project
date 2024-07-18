import psycopg2
from sentence_transformers import SentenceTransformer
import torch
import numpy as np

from dotenv import load_dotenv
import os

# Load environment variables from .env
load_dotenv()

# Accessing individual variables
db_host = os.getenv("DB_HOST")
db_name = os.getenv("DB_NAME")
db_user = os.getenv("DB_USER")
db_password = os.getenv("DB_PASSWORD")
db_port = os.getenv("DB_PORT")

# Set seeds for reproducibility
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)

# Establish the connection
conn = psycopg2.connect(
    host=db_host,
    dbname=db_name,
    user=db_user,
    password=db_password,
    port=db_port
)
cur = conn.cursor()

# Create the database table if not exists
create_table_query = '''
CREATE TABLE IF NOT EXISTS word_embeddings (
    id SERIAL PRIMARY KEY,
    word TEXT NOT NULL,
    embedding FLOAT8[]
);
'''
cur.execute(create_table_query)
conn.commit()

model = SentenceTransformer('all-MiniLM-L6-v2')


def get_words():
    with open('words.txt', 'r') as file:
        words = [line.strip() for line in file]
    letters = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ')
    words.extend(letters)
    return words

def fill_db(words):
    embeddings = model.encode(words)
    delete_query = "DELETE FROM word_embeddings"
    cur.execute(delete_query)
    for word, embedding in zip(words, embeddings):
        embedding_list = embedding.tolist()
        cur.execute(
            "INSERT INTO word_embeddings (word, embedding) VALUES (%s, %s)",
            (word, embedding_list)
        )
    conn.commit()

def get_number_of_rows_in_table(table_name='word_embeddings'):
    cur.execute(f"SELECT COUNT(*) FROM {table_name};")
    row_count = cur.fetchone()[0]
    return row_count


if __name__ == '__main__':
    words = get_words()
    fill_db(words)
    row_count = get_number_of_rows_in_table()
    print({"message": "Database filled", "row_count": row_count})