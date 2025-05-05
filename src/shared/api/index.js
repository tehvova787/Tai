import axios from 'axios';

// Создаем экземпляр axios с базовыми настройками
const api = axios.create({
  baseURL: process.env.REACT_APP_API_URL || 'https://api.luckytrain.io',
  timeout: 10000,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Добавляем перехватчик запросов для добавления токена авторизации
api.interceptors.request.use(
  (config) => {
    const token = localStorage.getItem('token') || sessionStorage.getItem('token');
    if (token) {
      config.headers.Authorization = `Bearer ${token}`;
    }
    return config;
  },
  (error) => Promise.reject(error)
);

// Добавляем перехватчик ответов для обработки ошибок
api.interceptors.response.use(
  (response) => response,
  (error) => {
    // Обработка ошибки авторизации
    if (error.response && error.response.status === 401) {
      // Очищаем токен и перенаправляем на страницу логина
      localStorage.removeItem('token');
      sessionStorage.removeItem('token');
      // В реальном приложении здесь будет редирект на страницу логина или событие
    }
    return Promise.reject(error);
  }
);

// API методы для работы с пользователями
export const userApi = {
  // Авторизация пользователя
  login: (email, password) => {
    return api.post('/auth/login', { email, password });
  },
  
  // Регистрация пользователя
  register: (userData) => {
    return api.post('/auth/register', userData);
  },
  
  // Выход из системы
  logout: () => {
    return api.post('/auth/logout');
  },
  
  // Получение профиля пользователя
  getProfile: () => {
    return api.get('/user/profile');
  },
  
  // Обновление профиля пользователя
  updateProfile: (profileData) => {
    return api.put('/user/profile', profileData);
  },
  
  // Смена пароля
  changePassword: (oldPassword, newPassword) => {
    return api.post('/user/change-password', { oldPassword, newPassword });
  },
};

// API методы для работы с кошельком
export const walletApi = {
  // Получение баланса
  getBalance: () => {
    return api.get('/wallet/balance');
  },
  
  // Получение истории транзакций
  getTransactions: (page = 1, limit = 10) => {
    return api.get(`/wallet/transactions?page=${page}&limit=${limit}`);
  },
  
  // Отправка транзакции
  sendTransaction: (transactionData) => {
    return api.post('/wallet/send', transactionData);
  },
  
  // Получение адреса кошелька
  getWalletAddress: () => {
    return api.get('/wallet/address');
  },
  
  // Резервное копирование ключа
  backupKey: (method) => {
    return api.post('/wallet/backup', { method });
  },
};

// API методы для работы с метавселенной
export const metaverseApi = {
  // Получение профиля персонажа
  getAvatarProfile: () => {
    return api.get('/metaverse/avatar');
  },
  
  // Обновление аватара
  updateAvatar: (avatarData) => {
    return api.put('/metaverse/avatar', avatarData);
  },
  
  // Получение инвентаря пользователя
  getInventory: () => {
    return api.get('/metaverse/inventory');
  },
  
  // Получение информации о локациях метавселенной
  getLocations: () => {
    return api.get('/metaverse/locations');
  },
};

// API методы для работы с AI ассистентом
export const aiApi = {
  // Отправка сообщения AI и получение ответа
  sendMessage: (message) => {
    return api.post('/ai/chat', { message });
  },
  
  // Получение истории чата с AI
  getChatHistory: () => {
    return api.get('/ai/chat/history');
  },
  
  // Очистка истории чата
  clearChatHistory: () => {
    return api.delete('/ai/chat/history');
  },
  
  // Получение информации о возможностях AI
  getAiCapabilities: () => {
    return api.get('/ai/capabilities');
  },
};

// API методы для работы с NFT
export const nftApi = {
  // Получение коллекции NFT пользователя
  getUserNfts: () => {
    return api.get('/nft/collection');
  },
  
  // Получение информации о конкретном NFT
  getNftDetails: (nftId) => {
    return api.get(`/nft/${nftId}`);
  },
  
  // Передача NFT другому пользователю
  transferNft: (nftId, recipientAddress) => {
    return api.post(`/nft/${nftId}/transfer`, { recipientAddress });
  },
};

// API методы для настроек
export const settingsApi = {
  // Получение настроек пользователя
  getSettings: () => {
    return api.get('/settings');
  },
  
  // Обновление настроек
  updateSettings: (settings) => {
    return api.put('/settings', settings);
  },
  
  // Проверка обновлений приложения
  checkForUpdates: () => {
    return api.get('/app/updates');
  },
};

export default {
  user: userApi,
  wallet: walletApi,
  metaverse: metaverseApi,
  ai: aiApi,
  nft: nftApi,
  settings: settingsApi,
}; 