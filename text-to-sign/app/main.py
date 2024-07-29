from flask import Flask, request,jsonify,send_file
import psycopg2
from sentence_transformers import SentenceTransformer
import torch
import numpy as np
import faiss
import language_tool_python
import spacy
import os
import time
from flask import request, render_template
from moviepy.editor import VideoFileClip, concatenate_videoclips
from dotenv import load_dotenv
import os
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Set seeds for reproducibility
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)


# Load environment variables from .env
load_dotenv()

# Accessing individual variables
db_host = os.getenv("DB_HOST")
db_name = os.getenv("DB_NAME")
db_user = os.getenv("DB_USER")
db_password = os.getenv("DB_PASSWORD")
db_port = os.getenv("DB_PORT")

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

# Initialize NLP tools
tool = language_tool_python.LanguageTool('en-US')
model = SentenceTransformer('all-MiniLM-L6-v2')
nlp = spacy.load('en_core_web_sm')


def get_words_and_embeddings_from_db():
    cur.execute("SELECT word, embedding FROM word_embeddings")
    records = cur.fetchall()
    words, embeddings = zip(*[(record[0], np.array(record[1])) for record in records])
    return words, embeddings


def semantic_search(query, top_k=5):
    query_embedding = model.encode([query])
    words, embeddings = get_words_and_embeddings_from_db()
    embeddings = np.array(embeddings)

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    distances, indices = index.search(query_embedding, top_k)

    similarities = 1 - distances / 2

    results = []
    threshold = 0.75


    print(f"Semantic Results of {query} :")
    for sim, idx in zip(similarities[0], indices[0]):
        if sim >= threshold:
            print(words[idx])
            results.append(words[idx])
    if not results:
        return [list(query.upper())]
    return results

def correct_english_grammar(text):
    matches = tool.check(text)
    corrected_text = language_tool_python.utils.correct(text, matches)
    return corrected_text



def get_lemmatized_words(sentence):
    doc = nlp(sentence)
    lemmas = [token.lemma_ for token in doc if token.pos_ not in ['AUX', 'ADP', 'SYM']]
    return lemmas

def semantic_search_multiword_glosses(lemmas):
    tokens = []
    for lemma in lemmas:
        results = semantic_search(lemma)
        tokens.append(results[0])
    return tokens

def concatenate_videos(video_paths, output_path):
    video_clips = []
    for video_path in video_paths:
        clip = VideoFileClip(video_path)
        video_clips.append(clip)

    concatenated_clip = concatenate_videoclips(video_clips)
    concatenated_clip.write_videofile(output_path, codec='libx264') 

def get_video_paths(tokens):
    video_paths = []
    for token in tokens:
        paths_to_check = []
        if isinstance(token, list):
            paths_to_check = [os.path.join('./videos', 'video_letters', letter + '.mp4') for letter in token]
        else:
            paths_to_check = [os.path.join('./videos', token + '.mp4')]

        for path in paths_to_check:
            if os.path.exists(path):
                video_paths.append(path)
            else:
                print("Path not found at", path)
    return video_paths
    

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process_sentence', methods=['POST'])
def process_sentence():
    sentence = request.form['sentence']
    output_path = './static/concatenated.mp4'

    start_time = time.time()
    corrected_sentence = correct_english_grammar(sentence)
    print("corrected: ", corrected_sentence)
    # fixed_sentence = fix_time_indicators(corrected_sentence)
    lemmas = get_lemmatized_words(corrected_sentence)
    print("lemmatized: ", lemmas)
    glosses = semantic_search_multiword_glosses(lemmas)
    print("glosses: ", glosses)
    gif_paths = get_video_paths(glosses)
    concatenate_videos(gif_paths, output_path)
    execution_time = time.time() - start_time
    print("output_path:", output_path)
    return render_template('ressult.html', sentence=corrected_sentence, execution_time=execution_time, output_path=output_path)

# @app.route('/process_sentence', methods=['POST'])
# def process_sentence():
#     sentence = request.form['sentence']
#     print(sentence)
#     output_path = './static/concatenated.mp4'

#     start_time = time.time()
#     corrected_sentence = correct_english_grammar(sentence)
#     print("corrected: ", corrected_sentence)
#     # fixed_sentence = fix_time_indicators(corrected_sentence)
#     lemmas = get_lemmatized_words(corrected_sentence)
#     print("lemmatized: ", lemmas)
#     glosses = semantic_search_multiword_glosses(lemmas)
#     print("glosses: ", glosses)
#     gif_paths = get_video_paths(glosses)
#     concatenate_videos(gif_paths, output_path)
#     # Existing processing logic...
#     # After creating the video file:

#     if os.path.exists(output_path):
#         return send_file(output_path, mimetype='video/mp4')
#     else:
#         return jsonify({'error': 'Video file not found'}), 404

if __name__ == '__main__':
    app.run(port=5001, debug=True, host='0.0.0.0')