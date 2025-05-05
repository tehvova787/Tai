import React, { useState } from 'react';
import {
  Box,
  Paper,
  Typography,
  Avatar,
  Grid,
  Divider,
  Button,
  Card,
  CardContent,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
  ListItemButton,
  TextField,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Alert,
  Snackbar
} from '@mui/material';
import AccountBalanceWalletIcon from '@mui/icons-material/AccountBalanceWallet';
import SwapHorizIcon from '@mui/icons-material/SwapHoriz';
import ImageIcon from '@mui/icons-material/Image';
import CalendarTodayIcon from '@mui/icons-material/CalendarToday';
import PersonIcon from '@mui/icons-material/Person';
import InventoryIcon from '@mui/icons-material/Inventory';
import EditIcon from '@mui/icons-material/Edit';
import ContentCopyIcon from '@mui/icons-material/ContentCopy';

const ProfilePage = () => {
  // Состояние для отображения модального окна редактирования профиля
  const [openEditDialog, setOpenEditDialog] = useState(false);
  // Состояние для уведомлений
  const [snackbar, setSnackbar] = useState({
    open: false,
    message: '',
    severity: 'success',
  });

  // Заглушка для данных профиля пользователя
  const [user, setUser] = useState({
    name: 'Пользователь',
    walletAddress: '0x1a2b3c4d5e6f7g8h9i0j',
    balance: '1,000 TON',
    joinDate: '01.01.2023',
    email: 'user@example.com',
    avatar: 'https://via.placeholder.com/150',
  });

  // Временное состояние для формы редактирования
  const [editUser, setEditUser] = useState({ ...user });

  // Обработчик открытия диалога редактирования
  const handleOpenEditDialog = () => {
    setEditUser({ ...user });
    setOpenEditDialog(true);
  };

  // Обработчик закрытия диалога редактирования
  const handleCloseEditDialog = () => {
    setOpenEditDialog(false);
  };

  // Обработчик сохранения изменений профиля
  const handleSaveProfile = () => {
    setUser({ ...editUser });
    setOpenEditDialog(false);
    setSnackbar({
      open: true,
      message: 'Профиль успешно обновлен',
      severity: 'success',
    });
  };

  // Обработчик изменения полей формы
  const handleChange = (e) => {
    setEditUser({
      ...editUser,
      [e.target.name]: e.target.value,
    });
  };

  // Обработчик копирования адреса кошелька
  const handleCopyWalletAddress = () => {
    navigator.clipboard.writeText(user.walletAddress);
    setSnackbar({
      open: true,
      message: 'Адрес кошелька скопирован в буфер обмена',
      severity: 'info',
    });
  };

  // Обработчик закрытия уведомления
  const handleCloseSnackbar = () => {
    setSnackbar({
      ...snackbar,
      open: false,
    });
  };

  return (
    <Box>
      <Grid container spacing={3}>
        {/* Левая колонка: информация о профиле */}
        <Grid item xs={12} md={4}>
          <Paper sx={{ p: 3, height: '100%' }}>
            <Box sx={{ display: 'flex', flexDirection: 'column', alignItems: 'center', mb: 3 }}>
              <Avatar
                src={user.avatar}
                alt={user.name}
                sx={{ width: 120, height: 120, mb: 2 }}
              />
              <Typography variant="h5" gutterBottom>
                {user.name}
              </Typography>
              <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
                <Typography variant="body2" color="text.secondary" sx={{ mr: 1 }}>
                  {user.walletAddress}
                </Typography>
                <IconButton
                  size="small"
                  onClick={handleCopyWalletAddress}
                  aria-label="Копировать адрес кошелька"
                >
                  <ContentCopyIcon fontSize="small" />
                </IconButton>
              </Box>
              <Button
                variant="outlined"
                startIcon={<EditIcon />}
                onClick={handleOpenEditDialog}
                sx={{ mt: 2 }}
              >
                Редактировать профиль
              </Button>
            </Box>

            <Divider sx={{ my: 2 }} />

            <List dense>
              <ListItem disablePadding>
                <ListItemIcon>
                  <CalendarTodayIcon />
                </ListItemIcon>
                <ListItemText
                  primary="Участник с"
                  secondary={user.joinDate}
                />
              </ListItem>
              <ListItem disablePadding sx={{ mt: 1 }}>
                <ListItemIcon>
                  <PersonIcon />
                </ListItemIcon>
                <ListItemText
                  primary="Email"
                  secondary={user.email}
                />
              </ListItem>
            </List>
          </Paper>
        </Grid>

        {/* Правая колонка: финансовая информация и действия */}
        <Grid item xs={12} md={8}>
          {/* Карточка баланса */}
          <Card sx={{ mb: 3 }}>
            <CardContent>
              <Grid container spacing={2} alignItems="center">
                <Grid item xs={12} sm={6}>
                  <Typography variant="h6" color="text.secondary" gutterBottom>
                    Баланс кошелька
                  </Typography>
                  <Typography variant="h3" component="div">
                    {user.balance}
                  </Typography>
                </Grid>
                <Grid item xs={12} sm={6} sx={{ display: 'flex', justifyContent: 'flex-end' }}>
                  <Button
                    variant="contained"
                    startIcon={<AccountBalanceWalletIcon />}
                    sx={{ mr: 1 }}
                  >
                    Пополнить
                  </Button>
                  <Button
                    variant="outlined"
                    startIcon={<SwapHorizIcon />}
                  >
                    Отправить
                  </Button>
                </Grid>
              </Grid>
            </CardContent>
          </Card>

          {/* Секция действий */}
          <Paper sx={{ p: 2 }}>
            <Typography variant="h6" gutterBottom>
              Действия
            </Typography>
            <Divider sx={{ mb: 2 }} />
            
            <List>
              <ListItemButton>
                <ListItemIcon>
                  <SwapHorizIcon />
                </ListItemIcon>
                <ListItemText
                  primary="Транзакции"
                  secondary="История ваших транзакций"
                />
              </ListItemButton>
              
              <Divider />
              
              <ListItemButton>
                <ListItemIcon>
                  <ImageIcon />
                </ListItemIcon>
                <ListItemText
                  primary="NFT Коллекция"
                  secondary="Ваши NFT и цифровые предметы"
                />
              </ListItemButton>
              
              <Divider />
              
              <ListItemButton>
                <ListItemIcon>
                  <PersonIcon />
                </ListItemIcon>
                <ListItemText
                  primary="Персонаж в метавселенной"
                  secondary="Управление вашим аватаром"
                />
              </ListItemButton>
              
              <Divider />
              
              <ListItemButton>
                <ListItemIcon>
                  <InventoryIcon />
                </ListItemIcon>
                <ListItemText
                  primary="Инвентарь"
                  secondary="Ваши предметы в метавселенной"
                />
              </ListItemButton>
            </List>
          </Paper>
        </Grid>
      </Grid>

      {/* Диалог редактирования профиля */}
      <Dialog open={openEditDialog} onClose={handleCloseEditDialog} maxWidth="sm" fullWidth>
        <DialogTitle>Редактирование профиля</DialogTitle>
        <DialogContent>
          <Box component="form" sx={{ mt: 1 }}>
            <TextField
              margin="normal"
              fullWidth
              label="Имя"
              name="name"
              value={editUser.name}
              onChange={handleChange}
            />
            <TextField
              margin="normal"
              fullWidth
              label="Email"
              name="email"
              type="email"
              value={editUser.email}
              onChange={handleChange}
            />
            <TextField
              margin="normal"
              fullWidth
              label="URL аватара"
              name="avatar"
              value={editUser.avatar}
              onChange={handleChange}
              helperText="Введите URL изображения для вашего аватара"
            />
          </Box>
        </DialogContent>
        <DialogActions>
          <Button onClick={handleCloseEditDialog}>Отмена</Button>
          <Button onClick={handleSaveProfile} variant="contained">Сохранить</Button>
        </DialogActions>
      </Dialog>

      {/* Уведомление (Snackbar) */}
      <Snackbar
        open={snackbar.open}
        autoHideDuration={6000}
        onClose={handleCloseSnackbar}
        anchorOrigin={{ vertical: 'bottom', horizontal: 'right' }}
      >
        <Alert onClose={handleCloseSnackbar} severity={snackbar.severity} sx={{ width: '100%' }}>
          {snackbar.message}
        </Alert>
      </Snackbar>
    </Box>
  );
};

export default ProfilePage; 