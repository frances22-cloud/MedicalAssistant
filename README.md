# NLP Chatbot: MediBot

MediBot is a Python-based NLP chatbot designed to act as a personal medical assistant. It uses a neural network model trained on intents to understand and respond to user queries related to medical information.

## Project Structure

- **`app.py`**: Python script containing the Flask web application and chatbot logic.
- **`index.html`**: HTML file for the chatbot user interface.
- **`chatbot_model.h5`**: Trained Keras model file for intent classification.
- **`classes.pkl`**: Pickled file containing classes of intents.
- **`intents.json`**: JSON file containing intents and responses.
- **`words.pkl`**: Pickled file containing processed words vocabulary.

## Installation

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/frances22-cloud/MedicalAssistant
   ```
   ```bash
   cd medibot
   ```

2. **Install Dependencies:**
   Ensure you have Python 3.x and pip installed. Then install required Python packages:

   ```bash
   pip install flask
   ```
   ```bash
   pip install nltk 
   ```
   ```bash
   pip install tensorflow
   ```
   ```bash
   pip install keras
   ```
   ```bash
   pip install numpy
   ```

3. **Download NLTK Data:**
   Run the following commands in Python to download necessary NLTK data:
   ```python
   import nltk
   nltk.download("punkt")
   nltk.download("wordnet")
   ```

## Running the Application

1. **Start the Flask App:**

   ```bash
   python app.py
   ```

   This will start the Flask development server.

2. **Access the Chatbot:**
   Open your web browser and go to `http://127.0.0.1:5000/` to interact with the chatbot interface.

## Usage

- Enter your message in the input box provided and press Enter.
- The chatbot will respond with relevant information based on the query using trained intents and responses.

## Additional Notes

- Ensure `chatbot_model.h5`, `classes.pkl`, `intents.json`, and `words.pkl` are correctly placed in the project directory.
- Customize intents and responses in `intents.json` to expand or modify the chatbot's capabilities.
- Modify the HTML and CSS in `index.html` to change the appearance and behavior of the chat interface.

---

Adjust the paths and details based on your actual project structure and requirements. This README provides a clear overview of how to set up, run, and use your NLP chatbot application.
