// Экспорт всех хранилищ состояния
import useUserStore from './userStore';
import useWalletStore from './walletStore';
import useAiStore from './aiStore';
import useSettingsStore from './settingsStore';

export {
  useUserStore,
  useWalletStore,
  useAiStore,
  useSettingsStore,
};

// Инициализация приложения - загрузка необходимых данных при запуске
export const initializeApp = async () => {
  try {
    // Проверяем авторизацию пользователя
    const isAuthenticated = await useUserStore.getState().checkAuth();
    
    // Если пользователь авторизован, загружаем данные
    if (isAuthenticated) {
      // Загружаем профиль пользователя
      await useUserStore.getState().fetchProfile();
      
      // Загружаем данные кошелька
      await useWalletStore.getState().loadWalletData();
      
      // Загружаем историю чата с AI
      await useAiStore.getState().fetchChatHistory();
    }
    
    // Загружаем и применяем настройки
    await useSettingsStore.getState().fetchSettings();
    useSettingsStore.getState().applySettings();
    
    return true;
  } catch (error) {
    console.error('Ошибка инициализации приложения:', error);
    return false;
  }
}; 