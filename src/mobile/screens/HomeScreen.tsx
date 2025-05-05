import React from 'react';
import { View, StyleSheet, ScrollView } from 'react-native';
import { Text, Card, Button, Appbar } from 'react-native-paper';

const HomeScreen = () => {
  return (
    <View style={styles.container}>
      <Appbar.Header>
        <Appbar.Content title="Lucky Train AI" />
      </Appbar.Header>
      
      <ScrollView style={styles.content}>
        <Card style={styles.card}>
          <Card.Cover source={{ uri: 'https://via.placeholder.com/400x200?text=Lucky+Train+AI' }} />
          <Card.Title title="Добро пожаловать в Lucky Train AI" />
          <Card.Content>
            <Text variant="bodyMedium">
              Интеллектуальный AI-ассистент для проекта Lucky Train на блокчейне TON.
            </Text>
          </Card.Content>
        </Card>
        
        <Card style={styles.card}>
          <Card.Title title="Новости проекта" />
          <Card.Content>
            <Text variant="bodyMedium">
              Последние обновления и события в экосистеме Lucky Train.
            </Text>
          </Card.Content>
        </Card>
        
        <Card style={styles.card}>
          <Card.Title title="Взаимодействие с AI-ассистентом" />
          <Card.Content>
            <Text variant="bodyMedium">
              Задавайте вопросы о проекте Lucky Train, токене, блокчейне TON и метавселенной.
            </Text>
            <Button 
              mode="contained" 
              style={styles.button}
              onPress={() => {}}
            >
              Начать чат
            </Button>
          </Card.Content>
        </Card>
      </ScrollView>
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#f5f5f5',
  },
  content: {
    padding: 16,
  },
  card: {
    marginBottom: 16,
  },
  button: {
    marginTop: 16,
    backgroundColor: '#5271FF',
  },
});

export default HomeScreen; 