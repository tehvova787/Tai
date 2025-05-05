import React, { useState } from 'react';
import { View, StyleSheet, FlatList, KeyboardAvoidingView, Platform } from 'react-native';
import { Text, TextInput, IconButton, Appbar, Surface, ActivityIndicator } from 'react-native-paper';

const ChatScreen = () => {
  const [message, setMessage] = useState('');
  const [messages, setMessages] = useState([
    { id: '1', text: 'Привет! Я Lucky Train AI. Чем я могу помочь?', fromUser: false },
  ]);
  const [loading, setLoading] = useState(false);

  const sendMessage = () => {
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
      // Ответ AI
      const aiResponse = {
        id: (Date.now() + 1).toString(),
        text: 'Это демонстрационный ответ от AI ассистента Lucky Train. Здесь будет интеграция с API бэкенда.',
        fromUser: false,
      };
      setMessages(prev => [...prev, aiResponse]);
      setLoading(false);
    }, 1000);
  };

  const renderMessage = ({ item }) => (
    <Surface
      style={[
        styles.messageBubble,
        item.fromUser ? styles.userMessage : styles.aiMessage,
      ]}
      elevation={1}
    >
      <Text style={styles.messageText}>{item.text}</Text>
    </Surface>
  );

  return (
    <View style={styles.container}>
      <Appbar.Header>
        <Appbar.BackAction onPress={() => {}} />
        <Appbar.Content title="Lucky Train AI Чат" />
      </Appbar.Header>

      <FlatList
        data={messages}
        renderItem={renderMessage}
        keyExtractor={item => item.id}
        style={styles.messageList}
        contentContainerStyle={styles.messageListContent}
      />

      {loading && (
        <View style={styles.loadingContainer}>
          <ActivityIndicator animating={true} color="#5271FF" />
        </View>
      )}

      <KeyboardAvoidingView
        behavior={Platform.OS === 'ios' ? 'padding' : undefined}
        keyboardVerticalOffset={Platform.OS === 'ios' ? 90 : 0}
      >
        <View style={styles.inputContainer}>
          <TextInput
            style={styles.input}
            value={message}
            onChangeText={setMessage}
            placeholder="Введите сообщение..."
            mode="outlined"
            multiline
          />
          <IconButton
            icon="send"
            mode="contained"
            containerColor="#5271FF"
            iconColor="white"
            size={24}
            onPress={sendMessage}
            disabled={message.trim() === ''}
          />
        </View>
      </KeyboardAvoidingView>
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#f5f5f5',
  },
  messageList: {
    flex: 1,
  },
  messageListContent: {
    padding: 16,
  },
  messageBubble: {
    maxWidth: '80%',
    padding: 12,
    borderRadius: 12,
    marginBottom: 12,
  },
  userMessage: {
    alignSelf: 'flex-end',
    backgroundColor: '#DCF8C6',
  },
  aiMessage: {
    alignSelf: 'flex-start',
    backgroundColor: 'white',
  },
  messageText: {
    fontSize: 16,
  },
  inputContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    padding: 8,
    backgroundColor: 'white',
  },
  input: {
    flex: 1,
    marginRight: 8,
  },
  loadingContainer: {
    padding: 8,
    alignItems: 'center',
  },
});

export default ChatScreen; 