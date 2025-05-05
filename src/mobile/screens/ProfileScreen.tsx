import React from 'react';
import { View, StyleSheet, ScrollView } from 'react-native';
import { Text, Avatar, List, Divider, Button, Appbar, Card } from 'react-native-paper';

const ProfileScreen = () => {
  // Заглушка для профиля пользователя
  const user = {
    name: 'Пользователь',
    walletAddress: '0x1a2b3c4d5e6f7g8h9i0j',
    balance: '1,000 TON',
    joinDate: '01.01.2023',
  };

  return (
    <View style={styles.container}>
      <Appbar.Header>
        <Appbar.Content title="Профиль" />
        <Appbar.Action icon="cog" onPress={() => {}} />
      </Appbar.Header>

      <ScrollView style={styles.content}>
        <View style={styles.profileHeader}>
          <Avatar.Image 
            size={80} 
            source={{ uri: 'https://via.placeholder.com/150' }} 
            style={styles.avatar}
          />
          <Text variant="headlineMedium">{user.name}</Text>
          <Text variant="bodySmall">{user.walletAddress}</Text>
          <Button 
            mode="outlined" 
            style={styles.editButton}
            onPress={() => {}}
          >
            Редактировать профиль
          </Button>
        </View>

        <Card style={styles.balanceCard}>
          <Card.Content>
            <Text variant="titleLarge">Баланс</Text>
            <Text variant="displaySmall">{user.balance}</Text>
            <Button 
              mode="contained" 
              style={styles.actionButton}
              onPress={() => {}}
            >
              Пополнить
            </Button>
          </Card.Content>
        </Card>

        <List.Section>
          <List.Subheader>Информация</List.Subheader>
          <List.Item 
            title="Участник с" 
            description={user.joinDate}
            left={props => <List.Icon {...props} icon="calendar" />}
          />
          <Divider />
          <List.Item 
            title="Транзакции" 
            description="История ваших транзакций"
            left={props => <List.Icon {...props} icon="swap-horizontal" />}
            onPress={() => {}}
            right={props => <List.Icon {...props} icon="chevron-right" />}
          />
          <Divider />
          <List.Item 
            title="NFT Коллекция" 
            description="Ваши NFT"
            left={props => <List.Icon {...props} icon="image-multiple" />}
            onPress={() => {}}
            right={props => <List.Icon {...props} icon="chevron-right" />}
          />
        </List.Section>

        <List.Section>
          <List.Subheader>Метавселенная</List.Subheader>
          <List.Item 
            title="Персонаж" 
            description="Ваш аватар в метавселенной"
            left={props => <List.Icon {...props} icon="account-outline" />}
            onPress={() => {}}
            right={props => <List.Icon {...props} icon="chevron-right" />}
          />
          <Divider />
          <List.Item 
            title="Инвентарь" 
            description="Ваши предметы"
            left={props => <List.Icon {...props} icon="briefcase-outline" />}
            onPress={() => {}}
            right={props => <List.Icon {...props} icon="chevron-right" />}
          />
        </List.Section>

        <View style={styles.spacing} />
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
    flex: 1,
  },
  profileHeader: {
    alignItems: 'center',
    padding: 20,
  },
  avatar: {
    marginBottom: 10,
  },
  editButton: {
    marginTop: 10,
  },
  balanceCard: {
    margin: 16,
    marginTop: 0,
  },
  actionButton: {
    marginTop: 10,
    backgroundColor: '#5271FF',
  },
  spacing: {
    height: 20,
  }
});

export default ProfileScreen; 