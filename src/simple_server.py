import os
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)
app.template_folder = os.path.join(os.path.dirname(__file__), 'web/templates')
app.static_folder = os.path.join(os.path.dirname(__file__), 'web/static')

@app.route('/')
def index():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Lucky Train AI</title>
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

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True) 