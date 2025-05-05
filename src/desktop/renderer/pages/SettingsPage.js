import React, { useState, useEffect } from 'react';
import {
  Box,
  Paper,
  Typography,
  Grid,
  Divider,
  Button,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
  ListItemSecondaryAction,
  Switch,
  TextField,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  FormHelperText,
  Card,
  CardContent,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogContentText,
  DialogActions,
  Snackbar,
  Alert,
  RadioGroup,
  FormControlLabel,
  Radio,
  Tooltip,
  IconButton
} from '@mui/material';
import DarkModeIcon from '@mui/icons-material/DarkMode';
import NotificationsIcon from '@mui/icons-material/Notifications';
import LanguageIcon from '@mui/icons-material/Language';
import LockIcon from '@mui/icons-material/Lock';
import BackupIcon from '@mui/icons-material/Backup';
import SecurityIcon from '@mui/icons-material/Security';
import InfoIcon from '@mui/icons-material/Info';
import UpdateIcon from '@mui/icons-material/Update';
import EmailIcon from '@mui/icons-material/Email';
import LogoutIcon from '@mui/icons-material/Logout';
import HelpIcon from '@mui/icons-material/Help';

const SettingsPage = () => {
  // Состояние для настроек
  const [settings, setSettings] = useState({
    darkMode: false,
    notifications: true,
    language: 'ru',
    autoBackup: true,
    twoFactorAuth: false,
    biometricAuth: false,
    autoUpdate: true,
    animationsEnabled: true,
    customApiEndpoint: '',
    privacyMode: 'balanced',
  });

  // Состояние для диалога выхода из аккаунта
  const [logoutDialogOpen, setLogoutDialogOpen] = useState(false);
  
  // Состояние для диалога резервного копирования
  const [backupDialogOpen, setBackupDialogOpen] = useState(false);
  
  // Состояние для снэкбаров (уведомлений)
  const [snackbar, setSnackbar] = useState({
    open: false,
    message: '',
    severity: 'success',
  });

  // Загрузка настроек при монтировании компонента
  useEffect(() => {
    // В реальном приложении здесь будет загрузка настроек из electronAPI или хранилища
    const loadSettings = async () => {
      try {
        // Пример использования electron API
        if (window.electronAPI) {
          const loadedSettings = await window.electronAPI.getSettings();
          setSettings(prevSettings => ({ ...prevSettings, ...loadedSettings }));
        }
      } catch (error) {
        console.error('Ошибка при загрузке настроек:', error);
      }
    };

    loadSettings();
  }, []);

  // Обработчик изменения настроек
  const handleSettingChange = (setting, value) => {
    setSettings((prevSettings) => ({
      ...prevSettings,
      [setting]: value,
    }));

    // В реальном приложении здесь будет сохранение настроек
    if (window.electronAPI) {
      window.electronAPI.saveSettings({
        ...settings,
        [setting]: value,
      });
    }

    setSnackbar({
      open: true,
      message: 'Настройки сохранены',
      severity: 'success',
    });
  };

  // Обработчик закрытия снэкбара
  const handleCloseSnackbar = () => {
    setSnackbar({
      ...snackbar,
      open: false,
    });
  };

  // Обработчик создания резервной копии
  const handleBackup = () => {
    setBackupDialogOpen(false);
    
    // В реальном приложении здесь будет логика создания резервной копии
    setTimeout(() => {
      setSnackbar({
        open: true,
        message: 'Резервная копия успешно создана',
        severity: 'success',
      });
    }, 1000);
  };

  // Обработчик выхода из аккаунта
  const handleLogout = () => {
    setLogoutDialogOpen(false);
    
    // В реальном приложении здесь будет логика выхода из аккаунта
    setTimeout(() => {
      setSnackbar({
        open: true,
        message: 'Вы вышли из аккаунта',
        severity: 'info',
      });
    }, 500);
  };

  // Обработчик проверки обновлений
  const handleCheckForUpdates = () => {
    // В реальном приложении здесь будет логика проверки обновлений
    if (window.electronAPI) {
      window.electronAPI.checkForUpdates()
        .then(result => {
          setSnackbar({
            open: true,
            message: result.hasUpdate 
              ? `Доступна новая версия: ${result.version}` 
              : 'У вас установлена последняя версия приложения',
            severity: result.hasUpdate ? 'info' : 'success',
          });
        })
        .catch(error => {
          setSnackbar({
            open: true,
            message: 'Ошибка при проверке обновлений',
            severity: 'error',
          });
        });
    } else {
      setSnackbar({
        open: true,
        message: 'Функция проверки обновлений недоступна',
        severity: 'warning',
      });
    }
  };

  return (
    <Box>
      <Typography variant="h4" gutterBottom>
        Настройки
      </Typography>
      <Divider sx={{ mb: 3 }} />

      <Grid container spacing={3}>
        {/* Основные настройки */}
        <Grid item xs={12} md={6}>
          <Paper sx={{ p: 2, height: '100%' }}>
            <Typography variant="h6" gutterBottom>
              Основные настройки
            </Typography>
            <Divider sx={{ mb: 2 }} />
            
            <List>
              <ListItem>
                <ListItemIcon>
                  <DarkModeIcon />
                </ListItemIcon>
                <ListItemText
                  primary="Темная тема"
                  secondary="Переключиться на темную тему приложения"
                />
                <ListItemSecondaryAction>
                  <Switch
                    edge="end"
                    checked={settings.darkMode}
                    onChange={(e) => handleSettingChange('darkMode', e.target.checked)}
                  />
                </ListItemSecondaryAction>
              </ListItem>
              
              <Divider variant="inset" component="li" />
              
              <ListItem>
                <ListItemIcon>
                  <NotificationsIcon />
                </ListItemIcon>
                <ListItemText
                  primary="Уведомления"
                  secondary="Получать уведомления о событиях"
                />
                <ListItemSecondaryAction>
                  <Switch
                    edge="end"
                    checked={settings.notifications}
                    onChange={(e) => handleSettingChange('notifications', e.target.checked)}
                  />
                </ListItemSecondaryAction>
              </ListItem>
              
              <Divider variant="inset" component="li" />
              
              <ListItem>
                <ListItemIcon>
                  <LanguageIcon />
                </ListItemIcon>
                <ListItemText
                  primary="Язык"
                  secondary="Выберите язык интерфейса"
                />
                <ListItemSecondaryAction sx={{ minWidth: 120 }}>
                  <FormControl variant="standard" size="small">
                    <Select
                      value={settings.language}
                      onChange={(e) => handleSettingChange('language', e.target.value)}
                      displayEmpty
                    >
                      <MenuItem value="ru">Русский</MenuItem>
                      <MenuItem value="en">English</MenuItem>
                    </Select>
                  </FormControl>
                </ListItemSecondaryAction>
              </ListItem>
            </List>
          </Paper>
        </Grid>

        {/* Настройки безопасности */}
        <Grid item xs={12} md={6}>
          <Paper sx={{ p: 2, height: '100%' }}>
            <Typography variant="h6" gutterBottom>
              Безопасность
            </Typography>
            <Divider sx={{ mb: 2 }} />
            
            <List>
              <ListItem button onClick={() => {}}>
                <ListItemIcon>
                  <LockIcon />
                </ListItemIcon>
                <ListItemText
                  primary="Изменить пароль"
                  secondary="Обновите пароль вашего аккаунта"
                />
              </ListItem>
              
              <Divider variant="inset" component="li" />
              
              <ListItem button onClick={() => setBackupDialogOpen(true)}>
                <ListItemIcon>
                  <BackupIcon />
                </ListItemIcon>
                <ListItemText
                  primary="Резервное копирование ключа"
                  secondary="Создайте резервную копию ключа кошелька"
                />
              </ListItem>
              
              <Divider variant="inset" component="li" />
              
              <ListItem>
                <ListItemIcon>
                  <SecurityIcon />
                </ListItemIcon>
                <ListItemText
                  primary="Двухфакторная аутентификация (2FA)"
                  secondary="Дополнительный уровень защиты для вашего аккаунта"
                />
                <ListItemSecondaryAction>
                  <Switch
                    edge="end"
                    checked={settings.twoFactorAuth}
                    onChange={(e) => handleSettingChange('twoFactorAuth', e.target.checked)}
                  />
                </ListItemSecondaryAction>
              </ListItem>
            </List>
          </Paper>
        </Grid>

        {/* Дополнительные настройки */}
        <Grid item xs={12}>
          <Paper sx={{ p: 2 }}>
            <Typography variant="h6" gutterBottom>
              Дополнительные настройки
            </Typography>
            <Divider sx={{ mb: 2 }} />
            
            <Grid container spacing={2}>
              <Grid item xs={12} md={6}>
                <FormControl component="fieldset" fullWidth sx={{ mb: 2 }}>
                  <Typography variant="subtitle1" gutterBottom>
                    Режим конфиденциальности
                  </Typography>
                  <RadioGroup
                    value={settings.privacyMode}
                    onChange={(e) => handleSettingChange('privacyMode', e.target.value)}
                  >
                    <FormControlLabel 
                      value="high" 
                      control={<Radio />} 
                      label="Высокий уровень приватности" 
                    />
                    <FormControlLabel 
                      value="balanced" 
                      control={<Radio />} 
                      label="Сбалансированный" 
                    />
                    <FormControlLabel 
                      value="performance" 
                      control={<Radio />} 
                      label="Приоритет производительности" 
                    />
                  </RadioGroup>
                </FormControl>
              </Grid>
              
              <Grid item xs={12} md={6}>
                <FormControl fullWidth sx={{ mb: 2 }}>
                  <TextField
                    label="Пользовательский API-endpoint"
                    value={settings.customApiEndpoint}
                    onChange={(e) => handleSettingChange('customApiEndpoint', e.target.value)}
                    placeholder="https://api.example.com"
                    helperText="Оставьте пустым для использования стандартного API"
                    variant="outlined"
                    size="small"
                  />
                </FormControl>
                
                <FormControl fullWidth sx={{ mb: 2 }}>
                  <FormControlLabel
                    control={
                      <Switch
                        checked={settings.autoUpdate}
                        onChange={(e) => handleSettingChange('autoUpdate', e.target.checked)}
                      />
                    }
                    label="Автоматически проверять обновления"
                  />
                </FormControl>
              </Grid>
            </Grid>
          </Paper>
        </Grid>

        {/* О приложении и версия */}
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 2 }}>
                <Typography variant="h6">
                  О приложении
                </Typography>
                <IconButton size="small" aria-label="информация">
                  <HelpIcon fontSize="small" />
                </IconButton>
              </Box>
              
              <Typography variant="body2" color="text.secondary" gutterBottom>
                Lucky Train AI Assistant - кроссплатформенное приложение для взаимодействия с проектом Lucky Train на блокчейне TON.
              </Typography>
              
              <Divider sx={{ my: 2 }} />
              
              <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <Typography variant="body1">
                  Версия: 1.0.0
                </Typography>
                <Button
                  startIcon={<UpdateIcon />}
                  onClick={handleCheckForUpdates}
                  size="small"
                >
                  Проверить обновления
                </Button>
              </Box>
            </CardContent>
          </Card>
        </Grid>
        
        {/* Контакты и выход */}
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Контакты и поддержка
              </Typography>
              
              <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                <EmailIcon sx={{ mr: 1 }} color="action" />
                <Typography variant="body2">
                  info@luckytrain.io
                </Typography>
              </Box>
              
              <Button
                variant="outlined"
                fullWidth
                sx={{ mb: 2 }}
              >
                Связаться с поддержкой
              </Button>
              
              <Divider sx={{ my: 2 }} />
              
              <Button
                variant="outlined"
                color="error"
                startIcon={<LogoutIcon />}
                fullWidth
                onClick={() => setLogoutDialogOpen(true)}
              >
                Выйти из аккаунта
              </Button>
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      {/* Диалог подтверждения выхода */}
      <Dialog
        open={logoutDialogOpen}
        onClose={() => setLogoutDialogOpen(false)}
      >
        <DialogTitle>Выход из аккаунта</DialogTitle>
        <DialogContent>
          <DialogContentText>
            Вы уверены, что хотите выйти из аккаунта?
          </DialogContentText>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setLogoutDialogOpen(false)}>
            Отмена
          </Button>
          <Button onClick={handleLogout} color="error">
            Выйти
          </Button>
        </DialogActions>
      </Dialog>

      {/* Диалог резервного копирования */}
      <Dialog
        open={backupDialogOpen}
        onClose={() => setBackupDialogOpen(false)}
      >
        <DialogTitle>Резервное копирование ключа</DialogTitle>
        <DialogContent>
          <DialogContentText>
            Выберите способ сохранения резервной копии вашего ключа.
            Храните резервную копию в безопасном месте!
          </DialogContentText>
          <FormControl component="fieldset" sx={{ mt: 2 }}>
            <RadioGroup defaultValue="file">
              <FormControlLabel value="file" control={<Radio />} label="Сохранить в файл" />
              <FormControlLabel value="qr" control={<Radio />} label="Показать QR-код" />
              <FormControlLabel value="seed" control={<Radio />} label="Показать seed-фразу" />
            </RadioGroup>
          </FormControl>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setBackupDialogOpen(false)}>
            Отмена
          </Button>
          <Button onClick={handleBackup} variant="contained">
            Создать резервную копию
          </Button>
        </DialogActions>
      </Dialog>

      {/* Уведомления */}
      <Snackbar
        open={snackbar.open}
        autoHideDuration={6000}
        onClose={handleCloseSnackbar}
        anchorOrigin={{ vertical: 'bottom', horizontal: 'center' }}
      >
        <Alert onClose={handleCloseSnackbar} severity={snackbar.severity} sx={{ width: '100%' }}>
          {snackbar.message}
        </Alert>
      </Snackbar>
    </Box>
  );
};

export default SettingsPage; 