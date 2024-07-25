# Flask App for Sign Language Recognition

This Flask app utilizes a trained model for sign language recognition using MediaPipe and TensorFlow/Keras.

## Setup Instructions

1. Clone the repository:

   ```bash
   git clone <repo-url>
   cd <repo-directory>
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

<!--  Download the trained model (`model_lstm_6_classes_0.98.h5`) and place it in the root directory of the project.-->

3. Run the Flask app:

   ```bash
   python app.py
   ```

4. Access the app in your browser at `http://localhost:3000`.

## Usage

1. Visit the app in your browser.
2. Upload a video file.
3. Wait for the app to process the video and display the predicted actions.
4. Interpret the predictions based on the recognized actions.

**Note:** The model currently recognizes 29 classes of actions.

## Dependencies

- Flask
- NumPy
- TensorFlow
- Matplotlib
- MediaPipe
- OpenCV (opencv-python-headless)

## File Structure

- `app.py`: Contains the Flask application code.
- `requirements.txt`: Lists all Python dependencies required for the app.
- `model_lstm_6_classes_0.98.h5`: Pre-trained LSTM model for action recognition.
- `index.html`: HTML template for the web interface.
