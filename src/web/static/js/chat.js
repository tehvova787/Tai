/**
 * Lucky Train AI Assistant - Chat functionality
 * 
 * This script handles the chat functionality for the web interface
 * including sending messages, receiving responses, and updating the UI.
 */

document.addEventListener('DOMContentLoaded', function() {
    const chatMessages = document.getElementById('chat-messages');
    const chatForm = document.getElementById('chat-form');
    const messageInput = document.getElementById('message-input');
    const clearChatBtn = document.getElementById('clear-chat-btn');
    const themeToggleBtn = document.getElementById('theme-toggle-btn');
    
    let sessionId = localStorage.getItem('session_id') || '';
    
    // Initialize autosize for textarea
    initializeAutosize();
    
    // Handle form submission
    chatForm.addEventListener('submit', function(e) {
        e.preventDefault();
        
        const message = messageInput.value.trim();
        if (!message) return;
        
        // Add user message to the chat
        addMessage('user', message);
        
        // Clear input
        messageInput.value = '';
        
        // Reset textarea height
        messageInput.style.height = 'auto';
        
        // Send message to API
        sendMessage(message);
    });
    
    // Clear chat button
    if (clearChatBtn) {
        clearChatBtn.addEventListener('click', function() {
            if (confirm('Are you sure you want to clear the chat history?')) {
                clearChat();
            }
        });
    }
    
    // Theme toggle button
    if (themeToggleBtn) {
        themeToggleBtn.addEventListener('click', function() {
            toggleTheme();
        });
    }
    
    // Load chat history from localStorage
    loadChatHistory();
    
    // Auto-resize textarea as user types
    function initializeAutosize() {
        messageInput.addEventListener('input', function() {
            this.style.height = 'auto';
            this.style.height = (this.scrollHeight) + 'px';
        });
    }
    
    // Add a message to the chat
    function addMessage(role, content, messageId) {
        const chatMessages = document.getElementById('chat-messages');
        if (!chatMessages) {
            console.error('Chat messages container not found');
            return;
        }

        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${role}`;
        
        if (messageId) {
            messageDiv.dataset.messageId = messageId;
        }
        
        const avatarDiv = document.createElement('div');
        avatarDiv.className = 'message-avatar';
        
        const avatarImg = document.createElement('img');
        avatarImg.src = role === 'user' 
            ? '/static/images/user-avatar.png' 
            : '/static/images/assistant-avatar.png';
        avatarImg.alt = role === 'user' ? 'User' : 'Assistant';
        
        avatarDiv.appendChild(avatarImg);
        
        const contentDiv = document.createElement('div');
        contentDiv.className = 'message-content';
        
        // Add the message content
        if (typeof content === 'string') {
            // Render markdown-like format
            const formattedContent = formatMarkdown(content);
            contentDiv.innerHTML = formattedContent;
        } else {
            contentDiv.textContent = 'Error: Invalid message content';
        }
        
        messageDiv.appendChild(avatarDiv);
        messageDiv.appendChild(contentDiv);
        
        // Add feedback buttons for assistant messages
        if (role === 'assistant' && messageId) {
            const feedbackDiv = document.createElement('div');
            feedbackDiv.className = 'message-feedback';
            feedbackDiv.innerHTML = `
                <button class="feedback-btn" data-rating="positive" title="This was helpful">
                    <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M14 9V5a3 3 0 0 0-3-3l-4 9v11h11.28a2 2 0 0 0 2-1.7l1.38-9a2 2 0 0 0-2-2.3zM7 22H4a2 2 0 0 1-2-2v-7a2 2 0 0 1-2-2h3"></path></svg>
                </button>
                <button class="feedback-btn" data-rating="negative" title="This was not helpful">
                    <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M10 15v4a3 3 0 0 0 3 3l4-9V2H5.72a2 2 0 0 0-2 1.7l-1.38 9a2 2 0 0 0 2 2.3zm7-13h2.67A2.31 2.31 0 0 1 22 4v7a2.31 2.31 0 0 1-2.33 2H17"></path></svg>
                </button>
            `;
            
            // Add event listeners for feedback buttons
            feedbackDiv.querySelectorAll('.feedback-btn').forEach(btn => {
                btn.addEventListener('click', function() {
                    const rating = this.dataset.rating;
                    sendFeedback(messageId, rating);
                    
                    // Show feedback confirmation
                    feedbackDiv.innerHTML = '<span class="feedback-thanks">Дякую за ваш відгук!</span>';
                });
            });
            
            contentDiv.appendChild(feedbackDiv);
        }
        
        chatMessages.appendChild(messageDiv);
        
        // Scroll to bottom with smooth animation
        setTimeout(() => {
            chatMessages.scrollTo({
                top: chatMessages.scrollHeight,
                behavior: 'smooth'
            });
        }, 100);
        
        // Save to chat history
        saveMessage(role, content, messageId);
    }
    
    // Format markdown-like text
    function formatMarkdown(text) {
        // Handle code blocks
        text = text.replace(/```(.*?)\n([\s\S]*?)```/g, '<pre><code>$2</code></pre>');
        
        // Handle inline code
        text = text.replace(/`([^`]+)`/g, '<code>$1</code>');
        
        // Handle bold
        text = text.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
        
        // Handle italic
        text = text.replace(/\*(.*?)\*/g, '<em>$1</em>');
        
        // Handle links
        text = text.replace(/\[([^\]]+)\]\(([^)]+)\)/g, '<a href="$2" target="_blank" rel="noopener noreferrer">$1</a>');
        
        // Handle lists
        text = text.replace(/^\s*-\s+(.*)$/gm, '<li>$1</li>');
        text = text.replace(/(<li>.*<\/li>)/gs, '<ul>$1</ul>');
        
        // Handle line breaks
        text = text.replace(/\n/g, '<br>');
        
        return text;
    }
    
    // Chat context storage
    let chatContext = {
        currentTopic: null,
        lastQuestion: null,
        conversationHistory: [],
        lastExchangeRate: null,
        lastRateUpdate: null,
        aiEnabled: false
    };

    // Backup answers for common questions
    const backupAnswers = {
        "инвестировать": "Для инвестиций в проект Lucky Train рекомендую:\n1. Приобрести токены LTT на бирже\n2. Участвовать в стейкинге\n3. Инвестировать в метавселенную\n\nТокен LTT обеспечивает:\n- Пассивный доход\n- Участие в управлении\n- Доступ к эксклюзивным функциям",
        "ltt": "LTT (Lucky Train Token) - это утилитарный токен проекта:\n- Торгуется на биржах TON\n- Используется для стейкинга\n- Даёт доступ к метавселенной\n- Обеспечивает пассивный доход",
        "метавселенная": "Метавселенная Lucky Train - это:\n- 3D виртуальный мир\n- Социальная платформа\n- Торговая площадка\n- Игровая экосистема\n\nОсобенности:\n- VR/AR поддержка\n- NFT интеграция\n- Социальное взаимодействие",
        "курс": "Актуальные курсы:\n- LTT/USDT: 0.15 USDT\n- LTT/TON: 0.05 TON\n\nКурсы обновляются каждые 5 минут",
        "помощь": "Я могу рассказать о:\n1. Инвестициях в проект\n2. Токене LTT\n3. Метавселенной\n4. Курсах валют\n\nЗадайте вопрос по любой теме!",
        "привет": "Здравствуйте! Я AI-ассистент проекта Lucky Train. Чем могу помочь?",
        "default": "Извините, я не совсем понял вопрос. Попробуйте переформулировать или напишите 'помощь' для получения списка тем."
    };

    // Keywords for better understanding
    const keywords = {
        "инвестировать": ["инвест", "купить", "вложить", "стейкинг", "доход"],
        "ltt": ["токен", "ltt", "монета", "крипта"],
        "метавселенная": ["мета", "vr", "3d", "виртуальный", "игра"],
        "курс": ["курс", "цена", "стоимость", "usd", "ton"],
        "помощь": ["помощь", "help", "что умеешь", "возможности"],
        "привет": ["привет", "здравствуй", "добрый день", "хай"]
    };

    // Process message and generate response
    async function processMessage(message) {
        const lowerMessage = message.toLowerCase();
        
        // Check for math operations
        if (lowerMessage.match(/^[\d\+\-\*\/\s\(\)]+$/)) {
            try {
                const result = eval(message.replace(/[^\d\+\-\*\/\(\)]/g, ''));
                return `Результат: ${result}`;
            } catch (error) {
                return 'Извините, не удалось вычислить выражение.';
            }
        }

        // Check for exchange rate queries
        if (lowerMessage.includes('курс') || lowerMessage.includes('usd') || 
            lowerMessage.includes('доллар') || lowerMessage.includes('usdt')) {
            return await handleExchangeRateQuery();
        }

        // Find matching topic based on keywords
        for (const [topic, words] of Object.entries(keywords)) {
            if (words.some(word => lowerMessage.includes(word))) {
                chatContext.currentTopic = topic;
                chatContext.lastQuestion = message;
                return backupAnswers[topic] || backupAnswers.default;
            }
        }

        // Check for follow-up questions
        if (chatContext.currentTopic && chatContext.lastQuestion) {
            const followUpResponses = {
                "инвестировать": {
                    "риск": "Инвестиции в LTT считаются умеренно рискованными:\n- Проект имеет рабочий продукт\n- Команда с опытом\n- Прозрачная экономика\n- Регулярные обновления",
                    "доход": "Доходность зависит от:\n1. Стейкинга (до 15% годовых)\n2. Торговли NFT\n3. Участия в метавселенной"
                },
                "ltt": {
                    "купить": "Купить LTT можно:\n1. На биржах TON\n2. Через наш сайт\n3. У других держателей",
                    "хранить": "Рекомендуемые кошельки:\n- Tonkeeper\n- Tonhub\n- MyTonWallet"
                },
                "метавселенная": {
                    "начать": "Чтобы начать:\n1. Создайте TON кошелек\n2. Купите LTT\n3. Зарегистрируйтесь в метавселенной",
                    "функции": "Основные функции:\n- Создание аватара\n- Торговля NFT\n- Социальное взаимодействие\n- Игровые активности"
                }
            };

            if (followUpResponses[chatContext.currentTopic]) {
                for (const [question, answer] of Object.entries(followUpResponses[chatContext.currentTopic])) {
                    if (lowerMessage.includes(question)) {
                        return answer;
                    }
                }
            }
        }

        // Default response
        return backupAnswers.default;
    }

    // Handle exchange rate queries
    async function handleExchangeRateQuery() {
        try {
            // Check if we need to update the rate (not more often than every 5 minutes)
            const now = new Date();
            if (!chatContext.lastRateUpdate || 
                (now - chatContext.lastRateUpdate) > 5 * 60 * 1000) {
                
                // In a real implementation, this would fetch from an API
                // For now, we'll use a mock rate
                chatContext.lastExchangeRate = 38.5 + Math.random() * 0.5;
                chatContext.lastRateUpdate = now;
            }

            return `Текущий курс USDT: ${chatContext.lastExchangeRate.toFixed(2)} UAH`;
        } catch (error) {
            console.error('Error fetching exchange rate:', error);
            return 'Извините, не удалось получить курс валют. Попробуйте позже.';
        }
    }

    // Send message to API
    async function sendMessage(message) {
        // Show typing indicator
        showTypingIndicator();
        
        // Add message to conversation history
        chatContext.conversationHistory.push({
            role: 'user',
            content: message,
            timestamp: new Date().toISOString()
        });

        try {
            // Process message using local logic
            const response = await processMessage(message);
            
            // Add response to chat
            addMessage('assistant', response);
            
            // Add response to conversation history
            chatContext.conversationHistory.push({
                role: 'assistant',
                content: response,
                timestamp: new Date().toISOString()
            });
        } catch (error) {
            console.error('Error processing message:', error);
            addErrorMessage('Извините, произошла ошибка при обработке сообщения.');
        } finally {
            removeTypingIndicator();
        }
    }
    
    // Show typing indicator
    function showTypingIndicator() {
        const indicatorDiv = document.createElement('div');
        indicatorDiv.className = 'message assistant typing-indicator';
        indicatorDiv.id = 'typing-indicator';
        
        const avatarDiv = document.createElement('div');
        avatarDiv.className = 'message-avatar';
        
        const thinkingRobot = document.createElement('div');
        thinkingRobot.className = 'thinking-robot';
        
        avatarDiv.appendChild(thinkingRobot);
        
        const contentDiv = document.createElement('div');
        contentDiv.className = 'message-content';
        contentDiv.innerHTML = `
            <div class="thinking-dots">
                <span></span>
                <span></span>
                <span></span>
            </div>
        `;
        
        indicatorDiv.appendChild(avatarDiv);
        indicatorDiv.appendChild(contentDiv);
        
        chatMessages.appendChild(indicatorDiv);
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }
    
    // Remove typing indicator
    function removeTypingIndicator() {
        const indicator = document.getElementById('typing-indicator');
        if (indicator) {
            indicator.remove();
        }
    }
    
    // Send feedback
    function sendFeedback(messageId, rating) {
        // Store feedback locally
        const feedback = {
            messageId: messageId,
            rating: rating === 'positive' ? 1 : 0,
            timestamp: new Date().toISOString(),
            context: chatContext.currentTopic,
            lastQuestion: chatContext.lastQuestion
        };
        
        // Get existing feedback from localStorage
        const feedbackHistory = JSON.parse(localStorage.getItem('feedback_history') || '[]');
        feedbackHistory.push(feedback);
        localStorage.setItem('feedback_history', JSON.stringify(feedbackHistory));
        
        console.log('Feedback stored locally:', feedback);
    }
    
    // Save a message to the chat history
    function saveMessage(role, content, messageId) {
        const history = JSON.parse(localStorage.getItem('chat_history') || '[]');
        
        history.push({
            role: role,
            content: content,
            timestamp: new Date().toISOString(),
            messageId: messageId
        });
        
        // Limit history length
        const maxHistory = 100;
        if (history.length > maxHistory) {
            history.splice(0, history.length - maxHistory);
        }
        
        localStorage.setItem('chat_history', JSON.stringify(history));
    }
    
    // Load chat history from localStorage
    function loadChatHistory() {
        const history = JSON.parse(localStorage.getItem('chat_history') || '[]');
        
        if (history.length > 0) {
            // Clear the initial assistant message
            chatMessages.innerHTML = '';
            
            // Add messages from history
            history.forEach(msg => {
                addMessage(msg.role, msg.content, msg.messageId);
            });
        }
    }
    
    // Clear chat history
    function clearChat() {
        // Clear UI
        chatMessages.innerHTML = '';
        
        // Add initial message
        addMessage('assistant', 'Привет! Я официальный AI-ассистент проекта Lucky Train. Чем я могу вам помочь?');
        
        // Clear localStorage
        localStorage.removeItem('chat_history');
        
        // Optionally reset session
        sessionId = '';
        localStorage.removeItem('session_id');
    }
    
    // Toggle between light and dark theme
    function toggleTheme() {
        const body = document.body;
        const isDark = body.classList.contains('theme-dark');
        
        if (isDark) {
            body.classList.remove('theme-dark');
            body.classList.add('theme-light');
            localStorage.setItem('theme', 'light');
        } else {
            body.classList.remove('theme-light');
            body.classList.add('theme-dark');
            localStorage.setItem('theme', 'dark');
        }
    }
    
    // Load saved theme
    function loadSavedTheme() {
        const savedTheme = localStorage.getItem('theme');
        if (savedTheme) {
            document.body.classList.remove('theme-light', 'theme-dark');
            document.body.classList.add(`theme-${savedTheme}`);
        }
    }
    
    // Load saved theme
    loadSavedTheme();
});

// Add a CSS class to make links with target="_blank" secure
document.addEventListener('DOMContentLoaded', function() {
    const externalLinks = document.querySelectorAll('a[target="_blank"]');
    
    externalLinks.forEach(link => {
        if (!link.hasAttribute('rel')) {
            link.setAttribute('rel', 'noopener noreferrer');
        }
    });
});

// Add error message to chat
function addErrorMessage(errorText) {
    const chatMessages = document.getElementById('chat-messages');
    if (!chatMessages) {
        console.error('Chat messages container not found');
        return;
    }

    const errorDiv = document.createElement('div');
    errorDiv.className = 'message error';
    
    const contentDiv = document.createElement('div');
    contentDiv.className = 'message-content error-content';
    contentDiv.innerHTML = `
        <p>
            <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                <circle cx="12" cy="12" r="10"></circle>
                <line x1="12" y1="8" x2="12" y2="12"></line>
                <line x1="12" y1="16" x2="12.01" y2="16"></line>
            </svg>
            ${errorText}
        </p>
        <button class="retry-button" onclick="retryConnection()">Повторити спробу</button>
    `;
    
    errorDiv.appendChild(contentDiv);
    chatMessages.appendChild(errorDiv);
    
    // Scroll to bottom with smooth animation
    setTimeout(() => {
        chatMessages.scrollTo({
            top: chatMessages.scrollHeight,
            behavior: 'smooth'
        });
    }, 100);
}

// Retry connection function
function retryConnection() {
    const chatMessages = document.getElementById('chat-messages');
    if (!chatMessages) {
        console.error('Chat messages container not found');
        return;
    }

    const lastMessage = document.querySelector('.message.user:last-child');
    if (lastMessage) {
        const messageContent = lastMessage.querySelector('.message-content').textContent;
        if (messageContent) {
            // Remove error message
            const errorMessage = document.querySelector('.message.error');
            if (errorMessage) {
                errorMessage.remove();
            }
            
            // Resend the last message
            const messageInput = document.getElementById('message-input');
            if (messageInput) {
                messageInput.value = messageContent;
                document.getElementById('chat-form').dispatchEvent(new Event('submit'));
            }
        }
    }
}