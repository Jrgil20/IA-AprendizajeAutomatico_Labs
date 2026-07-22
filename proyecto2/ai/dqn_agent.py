"""
Agente de aprendizaje por refuerzo con red neuronal profunda (Deep Q-Network).

La red es una CNN que, dada la posición del tablero desde la óptica de un
jugador, estima el valor Q de colocar una ficha en cada una de las 361
intersecciones. Entrenada por refuerzo (Q-learning con repetición de
experiencias y red objetivo), aprende a recomendar buenas jugadas.

Su salida se usa de dos formas:
  * Como agente directo (método act / get_action), útil para entrenar.
  * Como PLAYOUT POLICY del MCTS (método policy): durante la simulación del
    árbol, cada jugador coloca la ficha que la red recomienda en vez de una al
    azar. Ésa es la integración MCTS + Deep RL que pide el enunciado.
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from collections import deque
import random


class DQNAgent:
    def __init__(self, player_id, state_shape=(19, 19, 1), n_actions=361,
                 learning_rate=0.001, gamma=0.95, epsilon=1.0,
                 epsilon_min=0.01, epsilon_decay=0.995):
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
        """
        CNN para extraer patrones espaciales del tablero.
        Se construye con keras.Input (API funcional) para ser compatible tanto
        con Keras 2 como con Keras 3 (el que trae Colab): en Keras 3
        `InputLayer(input_shape=...)` ya no es válido.
        """
        inputs = keras.Input(shape=self.state_shape)
        x = keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
        x = keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = keras.layers.Flatten()(x)
        x = keras.layers.Dense(256, activation='relu')(x)
        outputs = keras.layers.Dense(self.n_actions)(x)
        return keras.Model(inputs=inputs, outputs=outputs)

    def update_target_network(self):
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action, reward, next_state, done):
        self.replay_buffer.append((state, action, reward, next_state, done))

    def decay_epsilon(self):
        """Reduce epsilon UNA vez por episodio (no por cada ficha)."""
        if self.epsilon > self.epsilon_min:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    # ------------------------------------------------------------------
    # Selección de jugada
    # ------------------------------------------------------------------
    def _q_values(self, board, player_id):
        state = board.get_state(player_id)
        state_batch = np.expand_dims(state, axis=0)
        # model(x) (llamada directa) es mucho más rápido que model.predict()
        # para lotes pequeños: predict() tiene gran sobrecarga por llamada.
        return self.model(state_batch, training=False).numpy()[0]

    def act(self, board, training=True):
        """Escoge una única acción (x, y) con Epsilon-Greedy y máscara de válidos."""
        valid_moves = board.get_valid_moves()
        if not valid_moves:
            return None

        valid_actions_flat = [x * board.size + y for x, y in valid_moves]

        if training and np.random.rand() <= self.epsilon:
            action_flat = random.choice(valid_actions_flat)
        else:
            q_values = self._q_values(board, self.player_id)
            masked_q_values = np.full(self.n_actions, -np.inf)
            masked_q_values[valid_actions_flat] = q_values[valid_actions_flat]
            action_flat = int(np.argmax(masked_q_values))

        x, y = divmod(action_flat, board.size)
        return (int(x), int(y))

    def recommend_move(self, board, player_id, candidates=None):
        """
        Devuelve la MEJOR jugada según la red para `player_id`, restringida a las
        casillas candidatas (poda de vecindad). Es la base de la playout policy.
        """
        if candidates is None:
            candidates = board.get_candidate_moves(radius=1)
        if not candidates:
            return None
        q_values = self._q_values(board, player_id)
        best, best_q = None, -np.inf
        for (x, y) in candidates:
            q = q_values[x * board.size + y]
            if q > best_q:
                best_q, best = q, (int(x), int(y))
        return best

    def policy(self, board, player_id, rng=None, epsilon=0.1):
        """
        Playout policy para el MCTS: casi siempre la jugada recomendada por la
        red; con probabilidad `epsilon` una candidata al azar (da variedad a las
        simulaciones). Firma compatible con MCTS: (board, player_id, rng).
        """
        candidates = board.get_candidate_moves(radius=1)
        if not candidates:
            return None
        if rng is not None and rng.random() < epsilon:
            return candidates[rng.randrange(len(candidates))]
        return self.recommend_move(board, player_id, candidates)

    def get_action(self, board, is_first_turn=False, training=False):
        """Interfaz de turno: retorna las N fichas (1 en el primer turno, si no 2)."""
        num_stones = 1 if is_first_turn else 2
        moves = []
        for _ in range(num_stones):
            move = self.act(board, training)
            if move:
                moves.append(move)
                board.grid[move[0], move[1]] = self.player_id  # ocupar temporal
        for m in moves:  # limpiar el rastro (main.py llamará a apply_move)
            board.grid[m[0], m[1]] = 0
        return moves

    # ------------------------------------------------------------------
    # Entrenamiento
    # ------------------------------------------------------------------
    def train_step(self, batch_size=32):
        if len(self.replay_buffer) < batch_size:
            return 0.0

        indices = np.random.choice(len(self.replay_buffer), batch_size, replace=False)
        batch = [self.replay_buffer[i] for i in indices]

        states, actions, rewards, next_states, dones = map(np.array, zip(*batch))
        rewards = rewards.astype(np.float32)
        dones = dones.astype(np.float32)

        next_Q_values = self.target_model(next_states, training=False).numpy()
        max_next_Q_values = np.max(next_Q_values, axis=1)
        target_Q_values = rewards + (1.0 - dones) * self.gamma * max_next_Q_values

        mask = tf.one_hot(actions, self.n_actions)

        with tf.GradientTape() as tape:
            all_Q_values = self.model(states)
            Q_values = tf.reduce_sum(all_Q_values * mask, axis=1, keepdims=True)
            target = tf.expand_dims(target_Q_values, axis=1)
            loss = tf.reduce_mean(self.loss_fn(target, Q_values))

        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

        return float(loss.numpy())

    # ------------------------------------------------------------------
    # Persistencia (formato .keras, el nativo de Keras 3)
    # ------------------------------------------------------------------
    def save(self, filename):
        self.model.save(filename)

    def load(self, filename):
        self.model = keras.models.load_model(filename)
        self.update_target_network()
