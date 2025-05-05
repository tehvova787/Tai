import React, { useState, useRef, useEffect } from 'react';
import { 
  Box, 
  Paper, 
  TextField, 
  Button, 
  Typography, 
  List, 
  ListItem, 
  ListItemText,
  Avatar,
  Divider,
  CircularProgress,
  IconButton
} from '@mui/material';
import SendIcon from '@mui/icons-material/Send';
import MicIcon from '@mui/icons-material/Mic';
import DeleteIcon from '@mui/icons-material/Delete';

const ChatPage = () => {
  const [message, setMessage] = useState('');
  const [messages, setMessages] = useState([
    { id: '1', text: 'Привет! Я Lucky Train AI. Чем я могу помочь?', fromUser: false },
  ]);
  const [loading, setLoading] = useState(false);
  const messagesEndRef = useRef(null);

  // Прокрутка чата вниз при добавлении новых сообщений
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  // Функция отправки сообщения
  const handleSendMessage = () => {
    if (message.trim() === '') return;
    
    // Добавляем сообщение пользователя
    const userMessage = {
      id: Date.now().toString(),
      text: message,
      fromUser: true,
    };
    
    setMessages([...messages, userMessage]);
    setMessage('');
    setLoading(true);
    
    // Имитация задержки ответа от AI
    setTimeout(() => {
      // В реальном приложении здесь будет вызов API
      const aiResponse = {
        id: (Date.now() + 1).toString(),
        text: 'Это демонстрационный ответ от AI ассистента Lucky Train. В реальном приложении здесь будет ответ от API бэкенда, использующего данные из базы знаний Lucky Train.',
        fromUser: false,
      };
      
      setMessages(prev => [...prev, aiResponse]);
      setLoading(false);
    }, 1500);
  };

  // Обработка нажатия Enter для отправки сообщения
  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  // Очистка истории чата
  const handleClearChat = () => {
    setMessages([
      { id: '1', text: 'Привет! Я Lucky Train AI. Чем я могу помочь?', fromUser: false },
    ]);
  };

  return (
    <Box sx={{ display: 'flex', flexDirection: 'column', height: '100%' }}>
      {/* Заголовок и кнопки управления */}
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
        <Typography variant="h5" component="h2">
          Чат с Lucky Train AI
        </Typography>
        <IconButton 
          color="error" 
          onClick={handleClearChat} 
          title="Очистить историю чата"
        >
          <DeleteIcon />
        </IconButton>
      </Box>
      
      <Divider sx={{ mb: 2 }} />
      
      {/* Основной контейнер чата */}
      <Paper 
        elevation={0}
        sx={{ 
          flexGrow: 1, 
          overflow: 'auto', 
          p: 2,
          backgroundColor: '#f5f5f5',
          mb: 2,
          borderRadius: 2,
        }}
      >
        <List>
          {messages.map((msg) => (
            <ListItem 
              key={msg.id} 
              sx={{ 
                justifyContent: msg.fromUser ? 'flex-end' : 'flex-start',
                mb: 1,
              }}
              disableGutters
            >
              <Box 
                sx={{ 
                  display: 'flex',
                  flexDirection: msg.fromUser ? 'row-reverse' : 'row',
                  alignItems: 'flex-start',
                  maxWidth: '80%',
                }}
              >
                {!msg.fromUser && (
                  <Avatar 
                    src="https://via.placeholder.com/40?text=AI" 
                    alt="Lucky Train AI"
                    sx={{ mr: msg.fromUser ? 0 : 1, ml: msg.fromUser ? 1 : 0 }}
                  />
                )}
                <Paper 
                  elevation={1}
                  sx={{
                    p: 2,
                    borderRadius: 2,
                    backgroundColor: msg.fromUser ? '#e3f2fd' : 'white',
                    maxWidth: '100%',
                  }}
                >
                  <ListItemText 
                    primary={msg.text}
                    primaryTypographyProps={{ 
                      component: 'div',
                      sx: { 
                        wordWrap: 'break-word',
                        whiteSpace: 'pre-wrap',
                      }
                    }}
                  />
                </Paper>
                {msg.fromUser && (
                  <Avatar 
                    sx={{ mr: msg.fromUser ? 0 : 1, ml: msg.fromUser ? 1 : 0 }}
                  />
                )}
              </Box>
            </ListItem>
          ))}
          {loading && (
            <ListItem sx={{ justifyContent: 'flex-start' }} disableGutters>
              <Box sx={{ display: 'flex', alignItems: 'center', ml: 5 }}>
                <CircularProgress size={24} color="primary" />
                <Typography variant="body2" color="text.secondary" sx={{ ml: 2 }}>
                  AI обрабатывает ваш запрос...
                </Typography>
              </Box>
            </ListItem>
          )}
          <div ref={messagesEndRef} />
        </List>
      </Paper>
      
      {/* Поле ввода сообщения */}
      <Paper 
        elevation={2}
        component="form"
        sx={{ 
          p: 1,
          display: 'flex',
          alignItems: 'center',
          borderRadius: 2,
        }}
      >
        <TextField
          fullWidth
          multiline
          maxRows={4}
          placeholder="Введите ваш вопрос..."
          value={message}
          onChange={(e) => setMessage(e.target.value)}
          onKeyPress={handleKeyPress}
          variant="outlined"
          size="small"
        />
        <IconButton color="primary" sx={{ ml: 1 }}>
          <MicIcon />
        </IconButton>
        <Button 
          variant="contained" 
          color="primary"
          endIcon={<SendIcon />}
          onClick={handleSendMessage}
          disabled={message.trim() === ''}
          sx={{ ml: 1 }}
        >
          Отправить
        </Button>
      </Paper>
    </Box>
  );
};

export default ChatPage; 