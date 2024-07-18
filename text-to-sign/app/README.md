### Setting Up PostgreSQL and Environment

1. **Install PostgreSQL**

   - Download and install PostgreSQL from official website.
   - Follow the installation instructions for your operating system.

2. **Open pgAdmin and Create Database**

   - Launch pgAdmin.
   - Connect to your PostgreSQL server.
   - Right-click on "Databases" and select "Create" > "Database..."
   - Name the database `SignVectorDB` (or your preferred name).
   - Click "Save" or "OK" to create the database.

3. **Create `.env` File**

   Create a `.env` file in the root directory of your project and add the following content:
   ```dotenv
   # PostgreSQL connection details
   DB_HOST=localhost
   DB_NAME=SignVectorDB
   DB_USER=postgres
   DB_PASSWORD=REPLACE_THIS
   DB_PORT=5432

  Replace **REPLACE_THIS** with your actual PostgreSQL password.

4. **Install Python Dependencies**
  - Open a terminal or command prompt and navigate to your project directory.
  - Run the following command to install dependencies from requirements.txt:
  ```bash
  pip install -r requirements.txt
  ```
5. **Download spaCy Model**
```bash
python -m spacy download en_core_web_sm
```
6. **Populate Database and Run Flask App**
   Populate the PostgreSQL database with word embeddings by running:
```bash
python fill_db.py
```
Start the Flask server to run the semantic search and video concatenation application:
```bash
python main.py
```
The Flask application will run locally at http://127.0.0.1:5000/.

### Using the Application
  Open a web browser and go to http://127.0.0.1:5000/.
  Enter a sentence into the provided input field.
  Submit the sentence to see it converted into American Sign Language (ASL) video.

