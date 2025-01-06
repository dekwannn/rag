from flask import Flask, render_template, request, jsonify
from main import rag  # Importing the 'rag' function from main.py

app = Flask(__name__)

# Route to serve the main page (your chatbot interface)
@app.route('/')
def index():
    return render_template('index.html')  # Serve the HTML file in 'templates'

# Route for processing user input from the chatbot
@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json.get('message')
    print("Input Pengguna:", user_input)  # Memeriksa input dari frontend

    if not user_input:
        return jsonify({'error': 'Pesan harus diisi'}), 400

    response = rag(user_input)
    print("Respons Bot:", response)  # Memeriksa respons dari fungsi rag

    return jsonify({'response': response})


if __name__ == '__main__':
    app.run(debug=True)
