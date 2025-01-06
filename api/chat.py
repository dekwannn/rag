from flask import Flask, request, jsonify
from main import rag  # Import fungsi 'rag' dari main.py

# Inisialisasi Flask
app = Flask(__name__)

# Rute untuk mengirimkan pesan dari frontend
@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json.get('message')
    print("Input Pengguna:", user_input)  # Memeriksa input dari frontend

    if not user_input:
        return jsonify({'error': 'Pesan harus diisi'}), 400

    response = rag(user_input)
    print("Respons Bot:", response)  # Memeriksa respons dari fungsi rag

    return jsonify({'response': response})

# Fungsi handler untuk Vercel (serverless)
def handler(request):
    with app.request_context(request):
        return app.full_dispatch_request()
