import React from 'react';
import { 
  Typography, 
  Grid, 
  Card, 
  CardContent, 
  CardMedia, 
  Button, 
  Box,
  Paper,
  Stack,
  Divider,
  CardActions
} from '@mui/material';
import ChatIcon from '@mui/icons-material/Chat';
import AccountBalanceWalletIcon from '@mui/icons-material/AccountBalanceWallet';
import SchoolIcon from '@mui/icons-material/School';
import VrpanoIcon from '@mui/icons-material/Vrpano';

const HomePage = () => {
  return (
    <Box sx={{ flexGrow: 1 }}>
      <Grid container spacing={3}>
        {/* Верхний баннер */}
        <Grid item xs={12}>
          <Paper
            elevation={2}
            sx={{
              p: 3,
              mb: 3,
              background: 'linear-gradient(45deg, #5271FF 30%, #4ECDC4 90%)',
              color: 'white',
              borderRadius: 2,
              position: 'relative',
              overflow: 'hidden',
            }}
          >
            <Grid container spacing={2} alignItems="center">
              <Grid item xs={12} md={8}>
                <Typography variant="h4" component="h1" gutterBottom>
                  Добро пожаловать в Lucky Train AI
                </Typography>
                <Typography variant="body1" paragraph>
                  Интеллектуальный AI-ассистент для проекта Lucky Train на блокчейне TON.
                  Получайте информацию о проекте, токене, блокчейне и метавселенной.
                </Typography>
                <Button 
                  variant="contained" 
                  color="secondary" 
                  size="large"
                  startIcon={<ChatIcon />}
                  sx={{ mt: 2 }}
                >
                  Начать общение с AI
                </Button>
              </Grid>
              <Grid item xs={12} md={4}>
                {/* Здесь может быть изображение или график */}
                <Box sx={{ display: { xs: 'none', md: 'block' } }}>
                  <img 
                    src="https://via.placeholder.com/300x200?text=Lucky+Train+AI" 
                    alt="Lucky Train AI"
                    style={{ width: '100%', borderRadius: '8px' }}
                  />
                </Box>
              </Grid>
            </Grid>
          </Paper>
        </Grid>

        {/* Карточки с функциями */}
        <Grid item xs={12} md={4}>
          <Card sx={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
            <CardMedia
              component="img"
              height="140"
              image="https://via.placeholder.com/400x200?text=AI+Assistant"
              alt="AI Assistant"
            />
            <CardContent sx={{ flexGrow: 1 }}>
              <Typography gutterBottom variant="h5" component="h2">
                AI Ассистент
              </Typography>
              <Typography>
                Задавайте вопросы о проекте Lucky Train, получайте актуальную информацию и помощь в использовании платформы.
              </Typography>
            </CardContent>
            <CardActions>
              <Button size="small" startIcon={<ChatIcon />}>Начать чат</Button>
            </CardActions>
          </Card>
        </Grid>

        <Grid item xs={12} md={4}>
          <Card sx={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
            <CardMedia
              component="img"
              height="140"
              image="https://via.placeholder.com/400x200?text=TON+Wallet"
              alt="TON Wallet"
            />
            <CardContent sx={{ flexGrow: 1 }}>
              <Typography gutterBottom variant="h5" component="h2">
                TON Кошелек
              </Typography>
              <Typography>
                Управляйте своими активами в сети TON, отправляйте и получайте токены, взаимодействуйте с блокчейном.
              </Typography>
            </CardContent>
            <CardActions>
              <Button size="small" startIcon={<AccountBalanceWalletIcon />}>Открыть кошелек</Button>
            </CardActions>
          </Card>
        </Grid>

        <Grid item xs={12} md={4}>
          <Card sx={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
            <CardMedia
              component="img"
              height="140"
              image="https://via.placeholder.com/400x200?text=Metaverse"
              alt="Metaverse"
            />
            <CardContent sx={{ flexGrow: 1 }}>
              <Typography gutterBottom variant="h5" component="h2">
                Метавселенная
              </Typography>
              <Typography>
                Исследуйте виртуальное пространство Lucky Train, взаимодействуйте с другими пользователями и объектами.
              </Typography>
            </CardContent>
            <CardActions>
              <Button size="small" startIcon={<VrpanoIcon />}>Войти в метавселенную</Button>
            </CardActions>
          </Card>
        </Grid>

        {/* Нижняя информационная секция */}
        <Grid item xs={12}>
          <Paper sx={{ p: 3, mt: 3, borderRadius: 2 }}>
            <Typography variant="h5" gutterBottom>
              Последние обновления
            </Typography>
            <Divider sx={{ mb: 2 }} />
            <Stack spacing={2}>
              <Box>
                <Typography variant="subtitle1" fontWeight="bold">
                  Обновление системы до версии 1.5.2
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  Улучшена производительность и добавлены новые функции для работы с блокчейном.
                </Typography>
              </Box>
              <Box>
                <Typography variant="subtitle1" fontWeight="bold">
                  Добавлена поддержка новых типов токенов
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  Теперь в системе поддерживаются NFT и новые стандарты токенов на блокчейне TON.
                </Typography>
              </Box>
              <Box>
                <Typography variant="subtitle1" fontWeight="bold">
                  Расширена база знаний
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  База знаний AI-ассистента пополнена новыми документами и информацией о проекте.
                </Typography>
              </Box>
            </Stack>
            <Box sx={{ mt: 2, display: 'flex', justifyContent: 'flex-end' }}>
              <Button startIcon={<SchoolIcon />}>
                Обучающие материалы
              </Button>
            </Box>
          </Paper>
        </Grid>
      </Grid>
    </Box>
  );
};

export default HomePage; 