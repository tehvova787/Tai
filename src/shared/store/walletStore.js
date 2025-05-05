import { create } from 'zustand';
import { walletApi } from '../api';

// Создание хранилища для работы с кошельком
const useWalletStore = create((set, get) => ({
  // Состояние кошелька
  balance: null,
  transactions: [],
  walletAddress: null,
  isLoading: false,
  error: null,
  currentPage: 1,
  totalPages: 1,
  
  // Получение баланса кошелька
  fetchBalance: async () => {
    set({ isLoading: true, error: null });
    try {
      const response = await walletApi.getBalance();
      set({
        balance: response.data.balance,
        isLoading: false,
      });
    } catch (error) {
      set({
        isLoading: false,
        error: error.response?.data?.message || 'Ошибка загрузки баланса',
      });
    }
  },
  
  // Получение адреса кошелька
  fetchWalletAddress: async () => {
    set({ isLoading: true, error: null });
    try {
      const response = await walletApi.getWalletAddress();
      set({
        walletAddress: response.data.address,
        isLoading: false,
      });
    } catch (error) {
      set({
        isLoading: false,
        error: error.response?.data?.message || 'Ошибка загрузки адреса кошелька',
      });
    }
  },
  
  // Получение транзакций пользователя
  fetchTransactions: async (page = 1, limit = 10) => {
    set({ isLoading: true, error: null, currentPage: page });
    try {
      const response = await walletApi.getTransactions(page, limit);
      set({
        transactions: response.data.transactions,
        totalPages: response.data.totalPages,
        isLoading: false,
      });
    } catch (error) {
      set({
        isLoading: false,
        error: error.response?.data?.message || 'Ошибка загрузки транзакций',
      });
    }
  },
  
  // Отправка транзакции
  sendTransaction: async (transactionData) => {
    set({ isLoading: true, error: null });
    try {
      const response = await walletApi.sendTransaction(transactionData);
      
      // Обновляем баланс после успешной транзакции
      await get().fetchBalance();
      
      // Обновляем список транзакций
      await get().fetchTransactions(1);
      
      set({ isLoading: false });
      return response.data;
    } catch (error) {
      set({
        isLoading: false,
        error: error.response?.data?.message || 'Ошибка при отправке транзакции',
      });
      return null;
    }
  },
  
  // Резервное копирование ключа
  backupKey: async (method) => {
    set({ isLoading: true, error: null });
    try {
      const response = await walletApi.backupKey(method);
      set({ isLoading: false });
      return response.data;
    } catch (error) {
      set({
        isLoading: false,
        error: error.response?.data?.message || 'Ошибка при создании резервной копии',
      });
      return null;
    }
  },
  
  // Загрузка всех данных кошелька (адрес, баланс, транзакции)
  loadWalletData: async () => {
    set({ isLoading: true, error: null });
    try {
      // Параллельно запускаем несколько запросов
      await Promise.all([
        get().fetchWalletAddress(),
        get().fetchBalance(),
        get().fetchTransactions(1),
      ]);
    } catch (error) {
      set({
        isLoading: false,
        error: error.message || 'Ошибка при загрузке данных кошелька',
      });
    }
  },
  
  // Сброс ошибки
  clearError: () => set({ error: null }),
  
  // Сброс состояния кошелька
  resetWallet: () => set({
    balance: null,
    transactions: [],
    walletAddress: null,
    isLoading: false,
    error: null,
    currentPage: 1,
    totalPages: 1,
  }),
}));

export default useWalletStore; 