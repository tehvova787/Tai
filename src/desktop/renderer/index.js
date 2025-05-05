import React from 'react';
import ReactDOM from 'react-dom/client';
import { ThemeProvider, createTheme } from '@mui/material/styles';
import CssBaseline from '@mui/material/CssBaseline';
import App from './App';

// Создаем корневой DOM-элемент для React
const root = ReactDOM.createRoot(document.getElementById('root'));

// Создаем тему MUI
const theme = createTheme({
  palette: {
    primary: {
      main: '#5271FF',
    },
    secondary: {
      main: '#4ECDC4',
    },
    background: {
      default: '#F5F5F5',
    },
  },
  typography: {
    fontFamily: 'Roboto, Arial, sans-serif',
  },
});

// Рендерим приложение
root.render(
  <React.StrictMode>
    <ThemeProvider theme={theme}>
      <CssBaseline /> {/* Нормализует стили */}
      <App />
    </ThemeProvider>
  </React.StrictMode>
); 