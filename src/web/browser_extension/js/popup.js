document.addEventListener('DOMContentLoaded', function() {
    const chatMessages = document.getElementById('chat-messages');
    const chatForm = document.getElementById('chat-form');
    const messageInput = document.getElementById('message-input');
    const clearChatBtn = document.getElementById('clear-chat-btn');
    const themeToggleBtn = document.getElementById('theme-toggle-btn');
    
    let sessionId = '';
    
    // Initialize extension
    init();
    
    // Set up event listeners
    chatForm.addEventListener('submit', handleChatSubmit);
    clearChatBtn.addEventListener('click', handleClearChat);
    themeToggleBtn.addEventListener('click', handleThemeToggle);
    messageInput.addEventListener('input', autoResizeTextarea);
    
    /**
     * Initialize the extension
     */
    async function init() {
        // Get session ID from storage
        const storage = await chrome.storage.local.get(['session_id', 'chat_history', 'theme']);
        sessionId = storage.session_id || '';
        
        // Apply saved theme
        if (storage.theme) {
            document.body.className = storage.theme === 'dark' ? 'dark-theme' : 'light-theme';
        }
        
        // Load chat history
        if (storage.chat_history && storage.chat_history.length > 0) {
            chatMessages.innerHTML = '';
            storage.chat_history.forEach(msg => {
                addMessageToChat(msg.role, msg.content);
            });
        }
    }
    
    /**
     * Handle chat form submission
     * @param {Event} e - Form submit event
     */
    function handleChatSubmit(e) {
        e.preventDefault();
        
        const message = messageInput.value.trim();
        if (!message) return;
        
        // Add user message to chat
        addMessageToChat('user', message);
        
        // Clear input
        messageInput.value = '';
        messageInput.style.height = 'auto';
        
        // Show loading indicator
        showTypingIndicator();
        
        // Send message to background script
        chrome.runtime.sendMessage({
            type: 'chat_request',
            data: {
                message: message,
                session_id: sessionId
            }
        }, function(response) {
            // Remove typing indicator
            removeTypingIndicator();
            
            if (response.error) {
                addErrorMessage(response.error);
                return;
            }
            
            // Save session ID
            if (response.session_id) {
                sessionId = response.session_id;
                chrome.storage.local.set({ session_id: sessionId });
            }
            
            // Add assistant message to chat
            addMessageToChat('assistant', response.response);
        });
    }
    
    /**
     * Add a message to the chat
     * @param {string} role - 'user' or 'assistant'
     * @param {string} content - Message content
     */
    function addMessageToChat(role, content) {
        const messageElement = document.createElement('div');
        messageElement.className = `message ${role}`;
        
        const contentElement = document.createElement('div');
        contentElement.className = 'message-content';
        
        // Format the content (handle markdown, links, etc.)
        contentElement.innerHTML = formatContent(content);
        
        messageElement.appendChild(contentElement);
        chatMessages.appendChild(messageElement);
        
        // Scroll to bottom
        chatMessages.scrollTop = chatMessages.scrollHeight;
        
        // Save to history
        saveMessageToHistory(role, content);
    }
    
    /**
     * Format message content
     * @param {string} content - Raw message content
     * @returns {string} - Formatted HTML content
     */
    function formatContent(content) {
        // Replace URLs with clickable links
        content = content.replace(
            /(https?:\/\/[^\s]+)/g, 
            '<a href="$1" target="_blank" rel="noopener noreferrer">$1</a>'
        );
        
        // Replace line breaks with <br>
        content = content.replace(/\n/g, '<br>');
        
        return content;
    }
    
    /**
     * Show typing indicator
     */
    function showTypingIndicator() {
        const indicator = document.createElement('div');
        indicator.className = 'message assistant typing';
        indicator.id = 'typing-indicator';
        
        const contentElement = document.createElement('div');
        contentElement.className = 'message-content';
        contentElement.innerHTML = '<div class="typing-dots"><span></span><span></span><span></span></div>';
        
        indicator.appendChild(contentElement);
        chatMessages.appendChild(indicator);
        
        // Scroll to bottom
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }
    
    /**
     * Remove typing indicator
     */
    function removeTypingIndicator() {
        const indicator = document.getElementById('typing-indicator');
        if (indicator) {
            indicator.remove();
        }
    }
    
    /**
     * Add an error message to the chat
     * @param {string} error - Error message
     */
    function addErrorMessage(error) {
        const messageElement = document.createElement('div');
        messageElement.className = 'message system error';
        
        const contentElement = document.createElement('div');
        contentElement.className = 'message-content';
        contentElement.innerHTML = `<p>Error: ${error}</p>`;
        
        messageElement.appendChild(contentElement);
        chatMessages.appendChild(messageElement);
        
        // Scroll to bottom
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }
    
    /**
     * Save message to history
     * @param {string} role - 'user' or 'assistant'
     * @param {string} content - Message content
     */
    async function saveMessageToHistory(role, content) {
        // Get current history
        const storage = await chrome.storage.local.get(['chat_history']);
        const history = storage.chat_history || [];
        
        // Add new message
        history.push({
            role: role,
            content: content,
            timestamp: new Date().toISOString()
        });
        
        // Limit history length
        const maxHistory = 50;
        if (history.length > maxHistory) {
            history.splice(0, history.length - maxHistory);
        }
        
        // Save updated history
        chrome.storage.local.set({ chat_history: history });
    }
    
    /**
     * Handle clear chat button click
     */
    function handleClearChat() {
        // Confirm deletion
        if (!confirm('Are you sure you want to clear the chat history?')) {
            return;
        }
        
        // Clear UI
        chatMessages.innerHTML = '';
        addMessageToChat('assistant', 'Привет! Я AI-ассистент проекта Lucky Train. Чем я могу помочь?');
        
        // Clear storage
        chrome.storage.local.remove(['chat_history', 'session_id']);
        sessionId = '';
    }
    
    /**
     * Handle theme toggle button click
     */
    function handleThemeToggle() {
        const isDark = document.body.classList.contains('dark-theme');
        
        if (isDark) {
            document.body.classList.remove('dark-theme');
            document.body.classList.add('light-theme');
            chrome.storage.local.set({ theme: 'light' });
        } else {
            document.body.classList.remove('light-theme');
            document.body.classList.add('dark-theme');
            chrome.storage.local.set({ theme: 'dark' });
        }
    }
    
    /**
     * Auto-resize textarea as user types
     */
    function autoResizeTextarea() {
        this.style.height = 'auto';
        this.style.height = (this.scrollHeight) + 'px';
    }
}); 