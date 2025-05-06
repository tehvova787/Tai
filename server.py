from flask import Flask, render_template, request, jsonify, send_from_directory
import os
import sys
import requests
import json
import time

# Create Flask app
app = Flask(__name__)

# Set template and static folders
base_dir = os.path.dirname(os.path.abspath(__file__))
app.template_folder = os.path.join(base_dir, 'src/web/templates')
app.static_folder = os.path.join(base_dir, 'src/web/static')

# Make sure static folder exists, if not, create a simple one
if not os.path.exists(app.static_folder):
    os.makedirs(app.static_folder, exist_ok=True)

# OpenAI API configuration
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY', '')
OPENAI_ORGANIZATION_ID = os.environ.get('OPENAI_ORGANIZATION_ID', '')

# Check if OpenAI API key is configured
if not OPENAI_API_KEY or OPENAI_API_KEY == 'your-api-key-here':
    print("Warning: OPENAI_API_KEY not properly configured. Chat functionality will not work.")
    print("To set the API key, use the following command:")
    print("  Windows: set OPENAI_API_KEY=your_actual_api_key")
    print("  Linux/Mac: export OPENAI_API_KEY=your_actual_api_key")

# OpenAI API endpoint
OPENAI_API_URL = "https://api.openai.com/v1/chat/completions"

# Chat API endpoint that calls OpenAI
@app.route('/api/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data provided"}), 400
            
        user_message = data.get('message')
        if not user_message:
            return jsonify({"error": "No message provided"}), 400
            
        session_id = data.get('session_id', f"session_{int(time.time())}")
        conversation_history = data.get('conversation_history', [])
        
        # Check if API key is properly configured
        if not OPENAI_API_KEY:
            return jsonify({
                "error": "OpenAI API key not configured",
                "response": "Извините, я не могу обработать этот запрос, так как API ключ OpenAI не настроен. Пожалуйста, обратитесь к администратору системы.",
                "message_id": f"error_{session_id}_{int(time.time())}",
                "session_id": session_id
            }), 200  # Return 200 to show error message to user rather than failing
        
        # Prepare the messages for OpenAI API
        messages = [
            {"role": "system", "content": "Ты ассистент проекта Lucky Train. Отвечай на украинском, русском или английском языке в зависимости от языка вопроса. Проект Lucky Train - это блокчейн проект на TON с метавселенной и собственным токеном LTT."}
        ]
        
        # Add conversation history if provided
        for msg in conversation_history:
            if isinstance(msg, dict) and 'role' in msg and 'content' in msg:
                messages.append({"role": msg["role"], "content": msg["content"]})
        
        # Add the current user message
        messages.append({"role": "user", "content": user_message})
        
        # Prepare headers
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {OPENAI_API_KEY}"
        }
        
        if OPENAI_ORGANIZATION_ID:
            headers["OpenAI-Organization"] = OPENAI_ORGANIZATION_ID
            
        # Prepare the request payload
        payload = {
            "model": "gpt-4o-mini",
            "messages": messages,
            "temperature": 0.7,
            "max_tokens": 1000
        }
        
        # Make the API request
        response = requests.post(OPENAI_API_URL, headers=headers, json=payload)
        response_data = response.json()
        
        if response.status_code != 200:
            print(f"OpenAI API Error: {response_data}")
            return jsonify({
                "error": "OpenAI API Error",
                "details": response_data.get("error", {}).get("message", "Unknown error")
            }), response.status_code
            
        # Extract the assistant's response
        assistant_message = response_data["choices"][0]["message"]["content"]
        
        return jsonify({
            "response": assistant_message,
            "message_id": f"chat_{session_id}_{int(time.time())}",
            "session_id": session_id
        })
        
    except Exception as e:
        print(f"Error in Chat API: {e}")
        return jsonify({"error": "Internal server error", "message": str(e)}), 500

# Simple home route
@app.route('/')
def index():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Lucky Train AI</title>
        <link rel="stylesheet" href="/static/css/tailwind.css">
        <style>
            body {
                font-family: Arial, sans-serif;
                margin: 0;
                padding: 0;
                display: flex;
                justify-content: center;
                align-items: center;
                min-height: 100vh;
                background-color: #f0f2f5;
            }
            .container {
                width: 80%;
                max-width: 800px;
                background-color: white;
                border-radius: 10px;
                box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
                padding: 20px;
            }
            h1 {
                color: #333;
                text-align: center;
            }
            .chat-box {
                height: 300px;
                border: 1px solid #ddd;
                border-radius: 5px;
                margin-bottom: 20px;
                padding: 10px;
                overflow-y: auto;
            }
            .input-area {
                display: flex;
            }
            input {
                flex: 1;
                padding: 10px;
                border: 1px solid #ddd;
                border-radius: 5px;
                margin-right: 10px;
            }
            button {
                padding: 10px 20px;
                background-color: #4a67e8;
                color: white;
                border: none;
                border-radius: 5px;
                cursor: pointer;
            }
            button:hover {
                background-color: #3954d4;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Lucky Train AI Ассистент</h1>
            <div class="chat-box" id="chatBox">
                <div><strong>Ассистент:</strong> Привет! Я AI-ассистент проекта Lucky Train. Чем я могу вам помочь?</div>
            </div>
            <div class="input-area">
                <input type="text" id="userInput" placeholder="Введите ваше сообщение...">
                <button onclick="sendMessage()">Отправить</button>
            </div>
        </div>

        <script>
            function sendMessage() {
                const userInput = document.getElementById('userInput');
                const chatBox = document.getElementById('chatBox');
                
                if (userInput.value.trim() === '') return;
                
                // Добавляем сообщение пользователя
                chatBox.innerHTML += `<div><strong>Вы:</strong> ${userInput.value}</div>`;
                
                // Простой ответ
                chatBox.innerHTML += `<div><strong>Ассистент:</strong> Спасибо за ваше сообщение! Это демонстрационная версия без полного функционала.</div>`;
                
                // Очищаем поле ввода
                userInput.value = '';
                
                // Прокручиваем чат вниз
                chatBox.scrollTop = chatBox.scrollHeight;
            }
            
            // Отправка сообщения при нажатии Enter
            document.getElementById('userInput').addEventListener('keypress', function(e) {
                if (e.key === 'Enter') {
                    sendMessage();
                }
            });
        </script>
    </body>
    </html>
    """

# Check if index.html exists and use it if available
@app.route('/index')
def html_index():
    try:
        return send_from_directory(base_dir, 'src/index.html')
    except:
        return "Index file not found. Using default page instead.", 404

# Homepage.html route
@app.route('/homepage')
def homepage():
    try:
        return send_from_directory(base_dir, 'homepage.html')
    except:
        return "Homepage file not found. Using default page instead.", 404

# LuckyTrainAI chat route
@app.route('/luckytrainai-chat')
def luckytrainai_chat():
    return render_template('luckytrainai-chat.html', 
                          title="LuckyTrainAI Чат",
                          welcome_message="Привет! Я LuckyTrainAI, чем могу помочь?")

# LuckyTrainAI interface route
@app.route('/luckytrainai')
def luckytrainai():
    return render_template('luckytrainai.html', 
                          title="LuckyTrainAI Интерфейс",
                          welcome_message="Добро пожаловать в интерфейс LuckyTrainAI!")

# Run the app
if __name__ == '__main__':
    port = 5000
    print(f"Starting server on http://localhost:{port}")
    # Disable .env loading to avoid null character issues
    app.config['ENV'] = 'production'
    app.run(host='0.0.0.0', port=port, debug=True, load_dotenv=False) 