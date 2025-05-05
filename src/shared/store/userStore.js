import { create } from 'zustand';
import { persist } from 'zustand/middleware';
import { userApi } from '../api';

// Создание хранилища пользователя с персистентностью (сохранением состояния)
const useUserStore = create(
  persist(
    (set, get) => ({
      // Состояние пользователя
      user: null,
      isAuthenticated: false,
      isLoading: false,
      error: null,
      token: null,

      // Авторизация пользователя
      login: async (email, password) => {
        set({ isLoading: true, error: null });
        try {
          const response = await userApi.login(email, password);
          const { user, token } = response.data;
          
          // Сохраняем токен в локальное хранилище
          localStorage.setItem('token', token);
          
          set({
            user,
            token,
            isAuthenticated: true,
            isLoading: false,
            error: null,
          });
          return true;
        } catch (error) {
          set({
            isLoading: false,
            error: error.response?.data?.message || 'Ошибка авторизации',
          });
          return false;
        }
      },

      // Регистрация пользователя
      register: async (userData) => {
        set({ isLoading: true, error: null });
        try {
          const response = await userApi.register(userData);
          const { user, token } = response.data;
          
          // Сохраняем токен в локальное хранилище
          localStorage.setItem('token', token);
          
          set({
            user,
            token,
            isAuthenticated: true,
            isLoading: false,
            error: null,
          });
          return true;
        } catch (error) {
          set({
            isLoading: false,
            error: error.response?.data?.message || 'Ошибка регистрации',
          });
          return false;
        }
      },

      // Выход из системы
      logout: async () => {
        set({ isLoading: true });
        try {
          await userApi.logout();
        } catch (error) {
          console.error('Ошибка при выходе:', error);
        } finally {
          // Очищаем локальное хранилище
          localStorage.removeItem('token');
          
          // Сбрасываем состояние
          set({
            user: null,
            token: null,
            isAuthenticated: false,
            isLoading: false,
            error: null,
          });
        }
      },

      // Получение профиля пользователя
      fetchProfile: async () => {
        set({ isLoading: true, error: null });
        try {
          const response = await userApi.getProfile();
          set({
            user: response.data,
            isLoading: false,
          });
        } catch (error) {
          set({
            isLoading: false,
            error: error.response?.data?.message || 'Ошибка загрузки профиля',
          });
        }
      },

      // Обновление профиля пользователя
      updateProfile: async (profileData) => {
        set({ isLoading: true, error: null });
        try {
          const response = await userApi.updateProfile(profileData);
          set({
            user: response.data,
            isLoading: false,
          });
          return true;
        } catch (error) {
          set({
            isLoading: false,
            error: error.response?.data?.message || 'Ошибка обновления профиля',
          });
          return false;
        }
      },

      // Смена пароля
      changePassword: async (oldPassword, newPassword) => {
        set({ isLoading: true, error: null });
        try {
          await userApi.changePassword(oldPassword, newPassword);
          set({ isLoading: false });
          return true;
        } catch (error) {
          set({
            isLoading: false,
            error: error.response?.data?.message || 'Ошибка смены пароля',
          });
          return false;
        }
      },

      // Проверка токена и автоматическая загрузка профиля
      checkAuth: async () => {
        const token = localStorage.getItem('token');
        if (!token) {
          return false;
        }

        set({ isLoading: true, token });
        try {
          const response = await userApi.getProfile();
          set({
            user: response.data,
            isAuthenticated: true,
            isLoading: false,
          });
          return true;
        } catch (error) {
          // Если токен недействителен, выполняем выход
          if (error.response?.status === 401) {
            localStorage.removeItem('token');
            set({
              user: null,
              token: null,
              isAuthenticated: false,
              isLoading: false,
            });
          } else {
            set({
              isLoading: false,
              error: error.response?.data?.message || 'Ошибка проверки авторизации',
            });
          }
          return false;
        }
      },

      // Сброс ошибки
      clearError: () => set({ error: null }),
    }),
    {
      name: 'user-storage', // Название для localStorage
      partialize: (state) => ({ token: state.token }), // Сохраняем только токен
    }
  )
);

export default useUserStore; 