"""
Ejecución local del agente de Connect6 (auto-juego de demostración).

Las negras las juega el agente principal MCTS + red neuronal + heurísticas.
Si existe un modelo entrenado (dqn_model.keras / .h5) se usa como playout policy
del MCTS; si no, el MCTS funciona igual con su playout heurístico (y NO requiere
TensorFlow, por eso la importación de la red es perezosa).

Las blancas las juega un agente aleatorio, sólo para ver una partida completa.
"""

from domain.board import Connect6Board
from ai.random_agent import RandomAgent
from ai.mcts_agent import MCTSAgent
from network.client import Connect6Client
import time
import sys
import os


def print_board(board):
    print("   " + " ".join(f"{i:2}" for i in range(board.size)))
    for i in range(board.size):
        row_str = f"{i:2} "
        for j in range(board.size):
            val = board.grid[i, j]
            row_str += " · " if val == 0 else (" X " if val == 1 else " O ")
        print(row_str)
    print()


def _load_black_agent(n_simulations):
    """Crea el agente de negras, cargando la red entrenada si existe."""
    for model_file in ("dqn_model.keras", "dqn_model.h5"):
        if os.path.exists(model_file):
            try:
                from ai.dqn_agent import DQNAgent  # import perezoso (necesita TensorFlow)
                dqn = DQNAgent(player_id=1, epsilon=0.0)
                dqn.load(model_file)
                print(f"[!] Modelo '{model_file}' cargado como playout del MCTS.")
                return MCTSAgent(player_id=1, dqn_agent=dqn, n_simulations=n_simulations)
            except Exception as e:
                print(f"[!] Hay un modelo '{model_file}' pero no se pudo cargar "
                      f"(¿TensorFlow no instalado? Python 3.14 aún no lo soporta).")
                print(f"    Detalle: {type(e).__name__}: {e}")
                print("    Se usará el playout heurístico del MCTS.")
                break
    print("[!] Sin red disponible: el MCTS usará su playout heurístico (no requiere TensorFlow).")
    print("    (Entrena la red en Colab con Entrenamiento_Connect6_Colab.ipynb para reforzar el playout.)")
    return MCTSAgent(player_id=1, dqn_agent=None, n_simulations=n_simulations)


def main(n_simulations=120, max_turns=200):
    print("=" * 44)
    print(" Agente Connect6 - MCTS (UCB1) + Deep RL + Heurísticas")
    print("=" * 44)

    board = Connect6Board()
    agent_black = _load_black_agent(n_simulations)
    agent_white = RandomAgent(player_id=2)

    client = Connect6Client(host='localhost', port=50051, agent_name="MCTS_DeepRL")
    try:
        client.connect()
    except Exception as e:
        print(f"Error al conectar al servidor: {e}")
        sys.exit(1)

    print("\nEstado inicial del tablero:")
    print_board(board)

    try:
        for turn in range(max_turns):
            is_first_turn = (turn == 0)

            # --- Turno de Negras (agente principal) ---
            moves_black = agent_black.get_action(board, is_first_turn=is_first_turn)
            if not moves_black:
                print("Empate: sin jugadas disponibles.")
                break
            print(f"Turno {turn + 1}: Negras (X) juegan en {moves_black}")
            board.apply_move(player_id=1, moves=moves_black)
            client.send_move(moves_black)
            print_board(board)
            if board.check_victory(player_id=1):
                print("¡Negras (X) han ganado!")
                break

            # --- Turno de Blancas (aleatorio) ---
            moves_white = agent_white.get_action(board, is_first_turn=False)
            if not moves_white:
                print("Empate: sin jugadas disponibles.")
                break
            print(f"Turno {turn + 1}: Blancas (O) juegan en {moves_white}")
            board.apply_move(player_id=2, moves=moves_white)
            print_board(board)
            if board.check_victory(player_id=2):
                print("¡Blancas (O) han ganado!")
                break
            time.sleep(0.2)

    except KeyboardInterrupt:
        print("\n[!] Partida interrumpida por el usuario.")
    finally:
        client.disconnect()
        print("Agente terminado limpiamente.")


if __name__ == "__main__":
    sims = int(sys.argv[1]) if len(sys.argv) > 1 else 120
    main(n_simulations=sims)
