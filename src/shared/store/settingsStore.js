import { create } from 'zustand';
import { persist } from 'zustand/middleware';
import { settingsApi } from '../api';

// Создание хранилища для настроек приложения
const useSettingsStore = create(
  persist(
    (set, get) => ({
      // Состояние настроек
      settings: {
        darkMode: false,
        language: 'ru',
        notifications: true,
        autoBackup: true,
        twoFactorAuth: false,
        biometricAuth: false,
        autoUpdate: true,
        animationsEnabled: true,
        customApiEndpoint: '',
        privacyMode: 'balanced',
      },
      isLoading: false,
      error: null,
      
      // Загрузка настроек с сервера
      fetchSettings: async () => {
        set({ isLoading: true, error: null });
        try {
          const response = await settingsApi.getSettings();
          set(state => ({
            settings: {
              ...state.settings,
              ...response.data,
            },
            isLoading: false,
          }));
        } catch (error) {
          set({
            isLoading: false,
            error: error.response?.data?.message || 'Ошибка загрузки настроек',
          });
        }
      },
      
      // Обновление настроек
      updateSettings: async (newSettings) => {
        set({ isLoading: true, error: null });
        try {
          // Объединяем текущие настройки с новыми
          const updatedSettings = {
            ...get().settings,
            ...newSettings,
          };
          
          // Сохраняем на сервере
          await settingsApi.updateSettings(updatedSettings);
          
          // Обновляем в хранилище
          set({
            settings: updatedSettings,
            isLoading: false,
          });
          
          return true;
        } catch (error) {
          set({
            isLoading: false,
            error: error.response?.data?.message || 'Ошибка обновления настроек',
          });
          return false;
        }
      },
      
      // Обновление отдельной настройки
      updateSetting: async (key, value) => {
        return await get().updateSettings({ [key]: value });
      },
      
      // Установка темной темы
      setDarkMode: async (enabled) => {
        await get().updateSetting('darkMode', enabled);
        
        // Если используется Electron, применяем тему к нативному приложению
        if (window.electronAPI && window.electronAPI.setNativeTheme) {
          window.electronAPI.setNativeTheme(enabled ? 'dark' : 'light');
        }
      },
      
      // Установка языка
      setLanguage: async (language) => {
        return await get().updateSetting('language', language);
      },
      
      // Проверка наличия обновлений приложения
      checkForUpdates: async () => {
        set({ isLoading: true, error: null });
        try {
          const response = await settingsApi.checkForUpdates();
          set({ isLoading: false });
          return response.data;
        } catch (error) {
          set({
            isLoading: false,
            error: error.response?.data?.message || 'Ошибка проверки обновлений',
          });
          return { hasUpdate: false };
        }
      },
      
      // Сброс настроек до значений по умолчанию
      resetSettings: async () => {
        const defaultSettings = {
          darkMode: false,
          language: 'ru',
          notifications: true,
          autoBackup: true,
          twoFactorAuth: false,
          biometricAuth: false,
          autoUpdate: true,
          animationsEnabled: true,
          customApiEndpoint: '',
          privacyMode: 'balanced',
        };
        
        return await get().updateSettings(defaultSettings);
      },
      
      // Применение настроек при запуске приложения
      applySettings: () => {
        const { settings } = get();
        
        // Применяем тему
        document.documentElement.setAttribute('data-theme', settings.darkMode ? 'dark' : 'light');
        
        // Применяем язык
        document.documentElement.setAttribute('lang', settings.language);
        
        // Если используется Electron, применяем настройки к нативному приложению
        if (window.electronAPI) {
          if (window.electronAPI.setNativeTheme) {
            window.electronAPI.setNativeTheme(settings.darkMode ? 'dark' : 'light');
          }
          
          if (window.electronAPI.setAutoUpdate) {
            window.electronAPI.setAutoUpdate(settings.autoUpdate);
          }
        }
      },
      
      // Сброс ошибки
      clearError: () => set({ error: null }),
    }),
    {
      name: 'settings-storage', // Название для localStorage
    }
  )
);

export default useSettingsStore; 