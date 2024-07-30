


<p align="center">
  <img src="https://github.com/user-attachments/assets/c160f8fe-827a-4eef-b20d-2bed7090095f" alt="Wavely Logo" />
</p>
  <h1 align="center">Wavely</h1>
  
  <h3 align="center">  Intelligent Collaboration Platform for Seamless Communication with Dedicated Support for Sign Language Accessibility</h3>
<hr>
<h2> Table of content </h2>
<li><a href="#0">Demo video</a></li>
<li><a href="#1">Overview</a> </li>
<li><a href="#2">Project Structure</a></li>
<li><a href="#3">Built Using</a></li>

<h2 id ="0">Demo video</h2>

https://github.com/user-attachments/assets/4f29ba7b-af81-4e11-8a37-5092f60f4abd

<h2 id ="1"> Overview </h2>
This project involves classifying sign language words from video inputs and using a pre-trained large language model (LLM) to form coherent sentences by understanding the context. The project also includes converting text into sign language video representations through an advanced pipeline that combines offline video processing with NLP techniques, real-time semantic search, and video concatenation techniques.

### Sign to Text
Our approach involves initially classifying words within sign language videos🖐️📹. Due to the differences in grammatical structures between sign language and the target language, we employed a pre-trained large language model (LLM) to understand the context of the input. This model helps in forming coherent sentences using the classified words, facilitated through few-shot prompting techniques🎯.

<p align="center">
  <img src="https://github.com/user-attachments/assets/5257fa66-4cb1-4f2b-8b0d-2494a74c1637" alt="Sign to text pipeline" />
</p>


This module utilizes `MediaPipe` for accurate keypoint estimation of facial landmarks, poses, and hand gestures to interpret sign language gestures. The pipeline includes:

<p align="center">
  <img src="https://github.com/user-attachments/assets/7f441c58-5319-4f71-b98d-595bfddce047" alt="Sign to words pipeline" />
</p>

Various deep learning models, including LSTM, Transformers, and others, were evaluated🔍📈. We ultimately selected the LSTM model for its consistent performance in avoiding overfitting and maintaining high accuracy🏆.

### Text to Sign

This module converts user-submitted sentences into concatenated video representations of sign language. The process is divided into two phases:

<p align="center">
  <img src="https://github.com/user-attachments/assets/a88161c3-48a3-4d89-991a-6aa330459480" alt="text to sign phases" />
</p>

-  🛠️ **`Offline Phase`**:
  - Convert videos into abstract skeleton representations.
  - Encode the processed words into high-dimensional vectors, facilitating efficient semantic search in the online phase.
-  🌐 **`Online Phase`**:
  - Retrieving and assembling skeleton videos into signing sequences using **NLP techniques**, **semantic search**, and **video processing tools**.
  
#### Offline Phase: 
  - 2000 videos of individuals signing various words are converted into abstract skeleton representations, ensuring privacy.
![mediapipe processing](https://github.com/user-attachments/assets/a8be3543-bd0d-48d4-a84d-f93c025ccec1)
  - A pre-trained model (all-MiniLM-L6-v2) is used to encode the 2000 words - we have processed videos for - into high dimensional vectors which are called `word embeddings`.
#### Online Phase:
- The input text is processed as follows to get the corresponding sign language video:

<p align="center">
  <img src="https://github.com/user-attachments/assets/936461db-eb46-4fa8-9833-0340120b98bd" alt="[online phase pipeline" />
</p>

##### 🔑 Key Functionalities

- ✅ **`Grammar correction`**: Automatically corrects grammatical errors in the text to ensure clear and accurate communication.
      
- 🔠 **`Lemmatization and Part-of-Speech (POS) Filtering`**: 
  - Lemmatization is the process of transforming a word into its canonical form. Unlike stemming, which merely truncates words to remove suffixes
  - Part-of-Speech (POS) Tagging: POS tagging involves assigning grammatical categories (such as noun, verb, adjective) to each word in a sentence.
          
- 🔍 **`Semantic search using FAISS (Facebook AI Similarity Search)`**:  to efficiently search and retrieve relevant information from high-dimensional data, enabling precise matching of sign language words with their text counterparts.
      
- 🎬 **`Video concatenation`**: Assembles individual sign language videos into a cohesive sequence, providing a seamless representation of text input in sign language

<h2 id ="2"> Project Structure </h2>

```plaintext

├── sign-to-text
│   ├── app
│   │   ├── templates
│   │   │   └── index.html
│   │   ├── .gitignore
│   │   ├── LSTM29.h5
│   │   ├── README.md
│   │   ├── app.py
│   │   └── requirements.txt
│   ├── notebooks
│   │   ├── Data_Collection.ipynb
│   │   ├── Sentence_Formation_Using_LLM.ipynb
│   │   ├── Sign_Lang_Translator_Word_Based.ipynb
│   │   ├── Real_Time_Testing_using_LSTM_Model.ipynb
│   │   └── Testing_Models.ipynb
│   ├── model-weights
│   │   ├── LSTM29.h5
│   │   └── Transformers29.h5
├── text-to-sign
│   ├── app
│   │   ├── static
│   │   ├── templates
│   │   │   ├── index.html
│   │   │   └── result.html
│   │   ├── README.md
│   │   ├── fill_db.py
│   │   ├── main.py
│   │   ├── requirements.txt
│   │   └── words.txt
│   ├── notebooks
│   │   └── Text_To_Sign_Conversion.ipynb
├── .gitignore
└── README.md
```

<h2 id ="3">Built using </h2>

- **`mediapipe`**: Cross-platform framework for building multimodal applied ML pipelines.
- **`tensorflow`**: Machine learning framework by Google
- **`torch`**: PyTorch deep learning library 
- **`torchaudio`**: Audio processing with PyTorch 
- **`torchvision`**: Computer vision tools for PyTorch
- **`numpy`**: Fundamental package for scientific computing with Python 
- **`sentence-transformers`**: Sentence embeddings using pre-trained Model
- **`opencv`**: OpenCV library with extra modules
- **`matplotlib`**: Plotting library for the Python programming language 
- **`spacy`**: Industrial-strength Natural Language Processing (NLP) library
- **`faiss-cpu`**: Facebook AI Similarity Search for efficient similarity search and clustering of dense vectors 
- **`Flask`**: Web framework for building the web application
- **`Flask-Cors`**: Handling Cross-Origin Resource Sharing (CORS)
- **`psycopg2-binary`**: PostgreSQL database adapter for Python 
- **`moviepy`**: Video editing with Python 
- **`python-dotenv`**: Read environment variables from a .env file
- **`language-tool-python`**: Grammar, style, and spelling checker
- **`transformers`**: Library for state-of-the-art natural language processing







