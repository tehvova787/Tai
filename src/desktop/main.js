const { app, BrowserWindow, ipcMain } = require('electron');
const path = require('path');
const Store = require('electron-store');

// Обработка запуска в режиме разработки
const isDev = process.env.NODE_ENV === 'development';

// Инициализация хранилища
const store = new Store();

// Создание основного окна приложения
function createWindow() {
  const mainWindow = new BrowserWindow({
    width: 1200,
    height: 800,
    minWidth: 800,
    minHeight: 600,
    webPreferences: {
      nodeIntegration: false,
      contextIsolation: true,
      preload: path.join(__dirname, 'preload.js'),
    },
    icon: path.join(__dirname, '../shared/assets/icons/icon.png')
  });

  // Загрузка HTML-файла
  if (isDev) {
    // В режиме разработки используем webpack dev-server
    mainWindow.loadURL('http://localhost:3000');
    mainWindow.webContents.openDevTools();
  } else {
    // В production загружаем собранный HTML файл
    mainWindow.loadFile(path.join(__dirname, 'renderer/index.html'));
  }

  // Событие закрытия окна
  mainWindow.on('closed', () => {
    mainWindow = null;
  });
}

// Создаем окно, когда Electron готов
app.whenReady().then(() => {
  createWindow();
  
  // На macOS обычно пересоздают окно, когда
  // пользователь кликает на иконку в dock
  app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) {
      createWindow();
    }
  });
});

// Выходим, когда все окна закрыты (Windows & Linux)
app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') {
    app.quit();
  }
});

// IPC обработчики для взаимодействия с рендерер-процессом

// Получение сохраненных настроек
ipcMain.handle('get-settings', () => {
  return store.get('settings') || {
    darkMode: false,
    language: 'ru',
    notifications: true
  };
});

// Сохранение настроек
ipcMain.handle('save-settings', (event, settings) => {
  store.set('settings', settings);
  return true;
});

// Обработчик для проверки обновлений
ipcMain.handle('check-for-updates', async () => {
  // Здесь будет логика проверки обновлений
  return { hasUpdate: false, version: app.getVersion() };
}); 