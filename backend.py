from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from gemini_1 import answer_with_context
import os
app = Flask(__name__)
CORS(app)

port = int(os.environ.get("PORT", 5000))

# API trả lời câu hỏi
@app.route('/ai_answer', methods=['POST'])
def ai_answer():
    data = request.get_json()

    if not data or "text" not in data:
        return jsonify({"error": "Missing 'text' field"}), 400

    user_question = data["text"].strip()
    if not user_question:
        return jsonify({"error": "Question cannot be empty"}), 400

    ai_response = answer_with_context(user_question)

    return jsonify({
        "question": user_question,
        "ai_answer": ai_response
    })

# Trang giao diện chatbot
@app.route("/")
def index():
    return render_template("chat.html")

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=port)
