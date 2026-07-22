"""
Entrenamiento por refuerzo de la red neuronal profunda (DQN) del agente.

La red aprende a recomendar jugadas jugando muchas partidas contra un agente
aleatorio (Q-learning con repetición de experiencias y red objetivo). El modelo
resultante (dqn_model.keras) se usa luego como PLAYOUT POLICY del MCTS.

Recomendado ejecutarlo en Google Colab con GPU (ver README_COLAB.md o el
notebook Entrenamiento_Connect6_Colab.ipynb): en 19x19 el entrenamiento es
costoso. Localmente requiere TensorFlow (que aún no da soporte a Python 3.14).
"""

import argparse
from domain.board import Connect6Board
from ai.dqn_agent import DQNAgent
from ai.random_agent import RandomAgent


def play_eval_game(agent_dqn, agent_random):
    """Una partida de evaluación (sin exploración). Devuelve 1 si gana la red."""
    board = Connect6Board()
    for turn in range(200):
        is_first = (turn == 0)
        black = agent_dqn.get_action(board, is_first_turn=is_first, training=False)
        if not black:
            return 0
        board.apply_move(1, black)
        if board.check_victory(1):
            return 1
        white = agent_random.get_action(board, is_first_turn=False)
        if not white:
            return 0
        board.apply_move(2, white)
        if board.check_victory(2):
            return 0
    return 0


def evaluate(agent_dqn, n_games=20):
    saved_eps = agent_dqn.epsilon
    agent_dqn.epsilon = 0.0
    rnd = RandomAgent(player_id=2)
    wins = sum(play_eval_game(agent_dqn, rnd) for _ in range(n_games))
    agent_dqn.epsilon = saved_eps
    return wins / n_games


def train_dqn(episodes=50, batch_size=32, target_update=5, model_path="dqn_model.keras"):
    print("=" * 44)
    print(" Entrenamiento del Agente DQN (Deep RL)")
    print("=" * 44)

    agent_dqn = DQNAgent(player_id=1, epsilon=1.0, epsilon_min=0.1, epsilon_decay=0.99)
    agent_random = RandomAgent(player_id=2)

    for episode in range(episodes):
        board = Connect6Board()
        done = False
        turn = 0
        total_reward = 0.0
        state = None
        action_flat = None

        while not done:
            is_first_turn = (turn == 0)

            # --- Turno del DQN (Negras) ---
            num_stones = 1 if is_first_turn else 2
            for _ in range(num_stones):
                state = board.get_state(player_id=1)
                move = agent_dqn.act(board, training=True)
                if move is None:
                    done = True
                    break
                action_flat = move[0] * board.size + move[1]
                reward, done, _ = board.step(player_id=1, move=move)
                next_state = board.get_state(player_id=1)
                agent_dqn.remember(state, action_flat, reward, next_state, done)
                total_reward += reward
                agent_dqn.train_step(batch_size)
                if done:
                    break
            if done:
                break

            # --- Turno del Agente Aleatorio (Blancas) ---
            moves_white = agent_random.get_action(board, is_first_turn=False)
            if not moves_white:
                done = True
                break
            for w_move in moves_white:
                _, done, _ = board.step(player_id=2, move=w_move)
                if done:  # si Blancas gana, castigamos la última jugada de Negras
                    next_state = board.get_state(player_id=1)
                    agent_dqn.remember(state, action_flat, -1.0, next_state, done)
                    total_reward -= 1.0
                    break
            turn += 1

        agent_dqn.decay_epsilon()  # decaimiento UNA vez por episodio
        if episode % target_update == 0:
            agent_dqn.update_target_network()

        print(f"Episodio {episode + 1}/{episodes} | Recompensa: {total_reward:+.1f} "
              f"| Epsilon: {agent_dqn.epsilon:.3f}")

    print(f"\nGuardando modelo en '{model_path}'...")
    agent_dqn.save(model_path)

    print("Evaluando el modelo contra el agente aleatorio...")
    win_rate = evaluate(agent_dqn, n_games=20)
    print(f"Tasa de victoria vs aleatorio: {win_rate * 100:.0f}%")
    print("Entrenamiento completado. Úsalo con main.py.")
    return agent_dqn


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entrena el DQN de Connect6.")
    parser.add_argument("--episodes", type=int, default=50,
                        help="Número de episodios (usa 5000-10000 en GPU).")
    parser.add_argument("--out", type=str, default="dqn_model.keras",
                        help="Ruta del modelo a guardar.")
    args = parser.parse_args()
    train_dqn(episodes=args.episodes, model_path=args.out)
