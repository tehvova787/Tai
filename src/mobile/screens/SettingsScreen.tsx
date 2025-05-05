import React, { useState } from 'react';
import { View, StyleSheet, ScrollView } from 'react-native';
import { Text, List, Switch, Divider, Button, Appbar, Dialog, Portal } from 'react-native-paper';

const SettingsScreen = () => {
  const [pushNotifications, setPushNotifications] = useState(true);
  const [darkMode, setDarkMode] = useState(false);
  const [biometricAuth, setBiometricAuth] = useState(false);
  const [confirmDialogVisible, setConfirmDialogVisible] = useState(false);

  return (
    <View style={styles.container}>
      <Appbar.Header>
        <Appbar.Content title="Настройки" />
      </Appbar.Header>

      <ScrollView style={styles.content}>
        <List.Section>
          <List.Subheader>Основные настройки</List.Subheader>
          
          <List.Item 
            title="Push-уведомления" 
            description="Включить уведомления о событиях"
            left={props => <List.Icon {...props} icon="bell-outline" />}
            right={() => 
              <Switch 
                value={pushNotifications} 
                onValueChange={setPushNotifications} 
                color="#5271FF"
              />
            }
          />
          
          <Divider />
          
          <List.Item 
            title="Темная тема" 
            description="Переключиться на темную тему"
            left={props => <List.Icon {...props} icon="theme-light-dark" />}
            right={() => 
              <Switch 
                value={darkMode} 
                onValueChange={setDarkMode} 
                color="#5271FF"
              />
            }
          />
          
          <Divider />
          
          <List.Item 
            title="Биометрическая аутентификация" 
            description="Использовать отпечаток пальца или Face ID"
            left={props => <List.Icon {...props} icon="fingerprint" />}
            right={() => 
              <Switch 
                value={biometricAuth} 
                onValueChange={setBiometricAuth} 
                color="#5271FF"
              />
            }
          />
        </List.Section>

        <List.Section>
          <List.Subheader>Безопасность</List.Subheader>
          
          <List.Item 
            title="Изменить пароль" 
            description="Обновите свой пароль"
            left={props => <List.Icon {...props} icon="lock-outline" />}
            onPress={() => {}}
          />
          
          <Divider />
          
          <List.Item 
            title="Резервное копирование ключа" 
            description="Создайте резервную копию вашего ключа кошелька"
            left={props => <List.Icon {...props} icon="content-save-outline" />}
            onPress={() => {}}
          />
          
          <Divider />
          
          <List.Item 
            title="Двухфакторная аутентификация" 
            description="Настройте 2FA для дополнительной защиты"
            left={props => <List.Icon {...props} icon="shield-outline" />}
            onPress={() => {}}
          />
        </List.Section>

        <List.Section>
          <List.Subheader>О приложении</List.Subheader>
          
          <List.Item 
            title="Версия приложения" 
            description="1.0.0"
            left={props => <List.Icon {...props} icon="information-outline" />}
          />
          
          <Divider />
          
          <List.Item 
            title="Проверить обновления" 
            description="Поиск новых версий приложения"
            left={props => <List.Icon {...props} icon="update" />}
            onPress={() => {}}
          />
          
          <Divider />
          
          <List.Item 
            title="Связаться с поддержкой" 
            description="Отправить сообщение команде поддержки"
            left={props => <List.Icon {...props} icon="email-outline" />}
            onPress={() => {}}
          />
        </List.Section>

        <View style={styles.buttonContainer}>
          <Button 
            mode="outlined" 
            style={styles.logoutButton}
            onPress={() => setConfirmDialogVisible(true)}
          >
            Выйти из аккаунта
          </Button>
        </View>

        <View style={styles.spacing} />
      </ScrollView>

      <Portal>
        <Dialog visible={confirmDialogVisible} onDismiss={() => setConfirmDialogVisible(false)}>
          <Dialog.Title>Выход из аккаунта</Dialog.Title>
          <Dialog.Content>
            <Text variant="bodyMedium">Вы уверены, что хотите выйти из аккаунта?</Text>
          </Dialog.Content>
          <Dialog.Actions>
            <Button onPress={() => setConfirmDialogVisible(false)}>Отмена</Button>
            <Button onPress={() => setConfirmDialogVisible(false)}>Выйти</Button>
          </Dialog.Actions>
        </Dialog>
      </Portal>
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
  buttonContainer: {
    padding: 16,
    alignItems: 'center',
  },
  logoutButton: {
    width: '100%',
    borderColor: '#ff3b30',
    borderWidth: 1,
  },
  spacing: {
    height: 20,
  }
});

export default SettingsScreen; 