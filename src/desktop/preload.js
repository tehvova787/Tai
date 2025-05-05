const { contextBridge, ipcRenderer } = require('electron');

// Предоставляем API рендереру, чтобы он мог взаимодействовать с главным процессом
contextBridge.exposeInMainWorld('electronAPI', {
  // Получение и сохранение настроек
  getSettings: () => ipcRenderer.invoke('get-settings'),
  saveSettings: (settings) => ipcRenderer.invoke('save-settings', settings),
  
  // Проверка обновлений
  checkForUpdates: () => ipcRenderer.invoke('check-for-updates'),
  
  // Операции с данными профиля пользователя
  getUserProfile: () => ipcRenderer.invoke('get-user-profile'),
  saveUserProfile: (profile) => ipcRenderer.invoke('save-user-profile', profile),
  
  // Операции блокчейна
  getWalletBalance: () => ipcRenderer.invoke('get-wallet-balance'),
  sendTransaction: (data) => ipcRenderer.invoke('send-transaction', data),
  
  // Интеграция с AI Assistant
  sendMessageToAI: (message) => ipcRenderer.invoke('send-message-to-ai', message),
  
  // Операции метавселенной
  getMetaverseProfile: () => ipcRenderer.invoke('get-metaverse-profile'),
  updateMetaverseAvatar: (data) => ipcRenderer.invoke('update-metaverse-avatar', data),
  
  // Слушатели событий для уведомлений
  onNotification: (callback) => {
    const channel = 'notification';
    const listener = (event, ...args) => callback(...args);
    
    ipcRenderer.on(channel, listener);
    return () => ipcRenderer.removeListener(channel, listener);
  },
  
  // Слушатель для обновлений блокчейна
  onBlockchainUpdate: (callback) => {
    const channel = 'blockchain-update';
    const listener = (event, ...args) => callback(...args);
    
    ipcRenderer.on(channel, listener);
    return () => ipcRenderer.removeListener(channel, listener);
  }
}); 