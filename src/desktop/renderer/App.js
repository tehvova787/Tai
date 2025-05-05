import React, { useState } from 'react';
import { Box, Drawer, AppBar, Toolbar, Typography, List, Divider, IconButton, 
         ListItem, ListItemIcon, ListItemText, Container, Paper } from '@mui/material';

// Импорт иконок
import MenuIcon from '@mui/icons-material/Menu';
import HomeIcon from '@mui/icons-material/Home';
import ChatIcon from '@mui/icons-material/Chat';
import AccountCircleIcon from '@mui/icons-material/AccountCircle';
import SettingsIcon from '@mui/icons-material/Settings';
import InfoIcon from '@mui/icons-material/Info';

// Импорт компонентов для каждой страницы
import HomePage from './pages/HomePage';
import ChatPage from './pages/ChatPage';
import ProfilePage from './pages/ProfilePage';
import SettingsPage from './pages/SettingsPage';

const drawerWidth = 240;

function App() {
  const [mobileOpen, setMobileOpen] = useState(false);
  const [currentPage, setCurrentPage] = useState('home');
  
  // Обработчик открытия/закрытия бокового меню
  const handleDrawerToggle = () => {
    setMobileOpen(!mobileOpen);
  };
  
  // Рендер текущей страницы
  const renderCurrentPage = () => {
    switch (currentPage) {
      case 'home':
        return <HomePage />;
      case 'chat':
        return <ChatPage />;
      case 'profile':
        return <ProfilePage />;
      case 'settings':
        return <SettingsPage />;
      default:
        return <HomePage />;
    }
  };
  
  // Содержимое бокового меню
  const drawer = (
    <div>
      <Toolbar sx={{ display: 'flex', justifyContent: 'center' }}>
        <Typography variant="h6" noWrap component="div">
          Lucky Train AI
        </Typography>
      </Toolbar>
      <Divider />
      <List>
        <ListItem button onClick={() => setCurrentPage('home')} selected={currentPage === 'home'}>
          <ListItemIcon>
            <HomeIcon color={currentPage === 'home' ? 'primary' : 'inherit'} />
          </ListItemIcon>
          <ListItemText primary="Главная" />
        </ListItem>
        
        <ListItem button onClick={() => setCurrentPage('chat')} selected={currentPage === 'chat'}>
          <ListItemIcon>
            <ChatIcon color={currentPage === 'chat' ? 'primary' : 'inherit'} />
          </ListItemIcon>
          <ListItemText primary="Чат с AI" />
        </ListItem>
        
        <ListItem button onClick={() => setCurrentPage('profile')} selected={currentPage === 'profile'}>
          <ListItemIcon>
            <AccountCircleIcon color={currentPage === 'profile' ? 'primary' : 'inherit'} />
          </ListItemIcon>
          <ListItemText primary="Профиль" />
        </ListItem>
        
        <ListItem button onClick={() => setCurrentPage('settings')} selected={currentPage === 'settings'}>
          <ListItemIcon>
            <SettingsIcon color={currentPage === 'settings' ? 'primary' : 'inherit'} />
          </ListItemIcon>
          <ListItemText primary="Настройки" />
        </ListItem>
      </List>
      <Divider />
      <List>
        <ListItem button>
          <ListItemIcon>
            <InfoIcon />
          </ListItemIcon>
          <ListItemText primary="О программе" />
        </ListItem>
      </List>
    </div>
  );

  return (
    <Box sx={{ display: 'flex', height: '100vh' }}>
      <AppBar
        position="fixed"
        sx={{
          width: { sm: `calc(100% - ${drawerWidth}px)` },
          ml: { sm: `${drawerWidth}px` },
        }}
      >
        <Toolbar>
          <IconButton
            color="inherit"
            aria-label="open drawer"
            edge="start"
            onClick={handleDrawerToggle}
            sx={{ mr: 2, display: { sm: 'none' } }}
          >
            <MenuIcon />
          </IconButton>
          <Typography variant="h6" noWrap component="div">
            {currentPage === 'home' && 'Главная'}
            {currentPage === 'chat' && 'Чат с AI'}
            {currentPage === 'profile' && 'Профиль'}
            {currentPage === 'settings' && 'Настройки'}
          </Typography>
        </Toolbar>
      </AppBar>
      
      <Box
        component="nav"
        sx={{ width: { sm: drawerWidth }, flexShrink: { sm: 0 } }}
        aria-label="mailbox folders"
      >
        {/* Мобильная версия */}
        <Drawer
          variant="temporary"
          open={mobileOpen}
          onClose={handleDrawerToggle}
          ModalProps={{
            keepMounted: true, // Лучшая производительность на мобильных устройствах
          }}
          sx={{
            display: { xs: 'block', sm: 'none' },
            '& .MuiDrawer-paper': { boxSizing: 'border-box', width: drawerWidth },
          }}
        >
          {drawer}
        </Drawer>
        
        {/* Десктопная версия */}
        <Drawer
          variant="permanent"
          sx={{
            display: { xs: 'none', sm: 'block' },
            '& .MuiDrawer-paper': { boxSizing: 'border-box', width: drawerWidth },
          }}
          open
        >
          {drawer}
        </Drawer>
      </Box>
      
      <Box
        component="main"
        sx={{ 
          flexGrow: 1, 
          p: 3, 
          width: { sm: `calc(100% - ${drawerWidth}px)` }, 
          height: '100%',
          overflow: 'auto',
          display: 'flex',
          flexDirection: 'column',
        }}
      >
        <Toolbar /> {/* Отступ под AppBar */}
        <Container maxWidth="lg" sx={{ flexGrow: 1, py: 2 }}>
          <Paper elevation={0} sx={{ height: '100%', p: 2 }}>
            {renderCurrentPage()}
          </Paper>
        </Container>
      </Box>
    </Box>
  );
}

export default App; 