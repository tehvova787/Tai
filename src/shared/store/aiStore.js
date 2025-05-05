import { create } from 'zustand';
import { persist } from 'zustand/middleware';
import { aiApi } from '../api';

// Создание хранилища для работы с AI-ассистентом
const useAiStore = create(
  persist(
    (set, get) => ({
      // Состояние чата с AI
      messages: [],
      isLoading: false,
      error: null,
      capabilities: null,
      
      // Отправка сообщения AI и получение ответа
      sendMessage: async (message) => {
        // Добавляем сообщение пользователя
        const userMessage = {
          id: Date.now().toString(),
          text: message,
          fromUser: true,
          timestamp: new Date().toISOString(),
        };
        
        set(state => ({
          messages: [...state.messages, userMessage],
          isLoading: true,
          error: null,
        }));
        
        try {
          const response = await aiApi.sendMessage(message);
          
          // Добавляем ответ от AI
          const aiMessage = {
            id: (Date.now() + 1).toString(),
            text: response.data.message,
            fromUser: false,
            timestamp: new Date().toISOString(),
          };
          
          set(state => ({
            messages: [...state.messages, aiMessage],
            isLoading: false,
          }));
          
          return aiMessage;
        } catch (error) {
          // Если не удалось получить ответ от сервера
          const errorMessage = {
            id: (Date.now() + 1).toString(),
            text: 'Извините, произошла ошибка при обработке вашего запроса. Пожалуйста, попробуйте позже.',
            fromUser: false,
            isError: true,
            timestamp: new Date().toISOString(),
          };
          
          set(state => ({
            messages: [...state.messages, errorMessage],
            isLoading: false,
            error: error.response?.data?.message || 'Ошибка взаимодействия с AI',
          }));
          
          return null;
        }
      },
      
      // Загрузка истории чата
      fetchChatHistory: async () => {
        set({ isLoading: true, error: null });
        try {
          const response = await aiApi.getChatHistory();
          set({
            messages: response.data,
            isLoading: false,
          });
        } catch (error) {
          set({
            isLoading: false,
            error: error.response?.data?.message || 'Ошибка загрузки истории чата',
          });
        }
      },
      
      // Очистка истории чата
      clearChatHistory: async () => {
        set({ isLoading: true, error: null });
        try {
          await aiApi.clearChatHistory();
          set({
            messages: [],
            isLoading: false,
          });
        } catch (error) {
          set({
            isLoading: false,
            error: error.response?.data?.message || 'Ошибка очистки истории чата',
          });
        }
      },
      
      // Получение информации о возможностях AI
      fetchAiCapabilities: async () => {
        set({ isLoading: true, error: null });
        try {
          const response = await aiApi.getAiCapabilities();
          set({
            capabilities: response.data,
            isLoading: false,
          });
        } catch (error) {
          set({
            isLoading: false,
            error: error.response?.data?.message || 'Ошибка загрузки информации о возможностях AI',
          });
        }
      },
      
      // Локальное добавление сообщения (например, для предварительного сообщения)
      addLocalMessage: (message) => {
        const newMessage = {
          id: Date.now().toString(),
          text: message,
          fromUser: false,
          timestamp: new Date().toISOString(),
        };
        
        set(state => ({
          messages: [...state.messages, newMessage],
        }));
        
        return newMessage;
      },
      
      // Сброс ошибки
      clearError: () => set({ error: null }),
      
      // Сброс состояния чата (без очистки истории на сервере)
      resetChatState: () => set({
        messages: [],
        isLoading: false,
        error: null,
      }),
    }),
    {
      name: 'ai-chat-storage', // Название для localStorage
      partialize: (state) => ({ messages: state.messages }), // Сохраняем только сообщения
    }
  )
);

export default useAiStore; 