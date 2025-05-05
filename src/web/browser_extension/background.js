// Background script for Lucky Train AI Assistant browser extension

// Handle installation
chrome.runtime.onInstalled.addListener((details) => {
  if (details.reason === 'install') {
    // Set default settings
    chrome.storage.local.set({
      theme: 'light',
      apiUrl: 'https://luckytrain.io/api',
      enableNotifications: true
    });
    
    // Open onboarding page
    chrome.tabs.create({
      url: 'https://luckytrain.io/extension-welcome'
    });
  }
});

// Listen for messages from popup or content scripts
chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  // Handle chat requests
  if (message.type === 'chat_request') {
    handleChatRequest(message.data)
      .then(response => sendResponse(response))
      .catch(error => sendResponse({ error: error.message }));
    return true; // Keep the message channel open for async response
  }
  
  // Handle settings updates
  if (message.type === 'update_settings') {
    chrome.storage.local.set(message.settings);
    sendResponse({ success: true });
    return false;
  }
});

/**
 * Handles chat requests by making API calls
 * @param {Object} data - The request data
 * @returns {Promise<Object>} - The response data
 */
async function handleChatRequest(data) {
  try {
    const settings = await chrome.storage.local.get(['apiUrl']);
    const apiUrl = settings.apiUrl || 'https://luckytrain.io/api';
    
    const response = await fetch(`${apiUrl}/chat`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({
        message: data.message,
        session_id: data.session_id
      })
    });
    
    if (!response.ok) {
      throw new Error(`API error: ${response.status}`);
    }
    
    const responseData = await response.json();
    return responseData;
  } catch (error) {
    console.error('Chat request error:', error);
    throw error;
  }
} 