import numpy as np
import tensorflow as tf
from tensorflow import keras
from collections import deque
import random

class DQNAgent:
    def __init__(self, player_id, state_shape=(19, 19, 1), n_actions=361, learning_rate=0.001, gamma=0.95, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995):
        self.player_id = player_id
        self.state_shape = state_shape
        self.n_actions = n_actions
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.learning_rate = learning_rate
        
        self.replay_buffer = deque(maxlen=10000)
        
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_network()
        
        self.loss_fn = keras.losses.Huber()
        self.optimizer = keras.optimizers.Adam(learning_rate=self.learning_rate)
        
    def _build_model(self):
        """Construye una red convolucional para extraer patrones espaciales del tablero."""
        model = keras.models.Sequential([
            keras.layers.InputLayer(input_shape=self.state_shape),
            keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            keras.layers.Flatten(),
            keras.layers.Dense(256, activation='relu'),
            keras.layers.Dense(self.n_actions)
        ])
        return model
        
    def update_target_network(self):
        self.target_model.set_weights(self.model.get_weights())
        
    def remember(self, state, action, reward, next_state, done):
        self.replay_buffer.append((state, action, reward, next_state, done))
        
    def act(self, board, training=True):
        """Escoge una única acción (x, y) utilizando Epsilon-Greedy y máscara de validación."""
        valid_moves = board.get_valid_moves()
        if not valid_moves:
            return None
            
        valid_actions_flat = [x * board.size + y for x, y in valid_moves]
        
        if training and np.random.rand() <= self.epsilon:
            # Exploración al azar sobre los válidos
            action_flat = random.choice(valid_actions_flat)
        else:
            # Explotación usando la DQN
            state = board.get_state(self.player_id)
            state_batch = np.expand_dims(state, axis=0)
            q_values = self.model.predict(state_batch, verbose=0)[0]
            
            # Enmascarar las acciones inválidas con -infinito
            masked_q_values = np.full(self.n_actions, -np.inf)
            masked_q_values[valid_actions_flat] = q_values[valid_actions_flat]
            action_flat = np.argmax(masked_q_values)
            
        x, y = divmod(action_flat, board.size)
        return (x, y)

    def get_action(self, board, is_first_turn=False, training=False):
        """Interfaz para jugar en el main.py que retorna las N coordenadas juntas."""
        num_stones = 1 if is_first_turn else 2
        moves = []
        for _ in range(num_stones):
            move = self.act(board, training)
            if move:
                moves.append(move)
                # Ocupamos la celda temporalmente para no repetir la coordenada
                board.grid[move[0], move[1]] = self.player_id
                
        # Limpiamos el rastro temporal (main.py llamará a apply_move)
        for m in moves:
            board.grid[m[0], m[1]] = 0
            
        return moves

    def train_step(self, batch_size=32):
        if len(self.replay_buffer) < batch_size:
            return 0
            
        indices = np.random.choice(len(self.replay_buffer), batch_size, replace=False)
        batch = [self.replay_buffer[i] for i in indices]
        
        states, actions, rewards, next_states, dones = map(np.array, zip(*batch))
        
        next_Q_values = self.target_model.predict(next_states, verbose=0)
        max_next_Q_values = np.max(next_Q_values, axis=1)
        target_Q_values = rewards + (1 - dones) * self.gamma * max_next_Q_values
        
        mask = tf.one_hot(actions, self.n_actions)
        
        with tf.GradientTape() as tape:
            all_Q_values = self.model(states)
            Q_values = tf.reduce_sum(all_Q_values * mask, axis=1, keepdims=True)
            target_Q_values = tf.expand_dims(target_Q_values, axis=1)
            loss = tf.reduce_mean(self.loss_fn(target_Q_values, Q_values))
            
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            
        return loss.numpy()
        
    def save(self, filename):
        self.model.save(filename)
        
    def load(self, filename):
        self.model = keras.models.load_model(filename)
        self.update_target_network()
