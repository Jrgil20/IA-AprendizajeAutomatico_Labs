from domain.board import Connect6Board
from ai.random_agent import RandomAgent
from ai.dqn_agent import DQNAgent
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
            if val == 0:
                row_str += " · "
            elif val == 1:
                row_str += " X "
            elif val == 2:
                row_str += " O "
        print(row_str)
    print()

def main():
    print("="*40)
    print(" Iniciando Agente Connect6 - Fase 1")
    print("="*40)
    
    # 1. Inicializar la lógica de dominio y los agentes
    board = Connect6Board()
    
    if os.path.exists("dqn_model.h5"):
        print("[!] Modelo dqn_model.h5 encontrado. Cargando Agente DQN...")
        agent_black = DQNAgent(player_id=1, epsilon=0.0) # Modo inferencia pura (sin exploración)
        agent_black.load("dqn_model.h5")
    else:
        print("[!] No se encontró dqn_model.h5 (Debes ejecutar train.py). Usando RandomAgent.")
        agent_black = RandomAgent(player_id=1) # Negras (X)
        
    agent_white = RandomAgent(player_id=2) # Blancas (O)
    
    # 2. Inicializar cliente gRPC (simulado)
    client = Connect6Client(host='localhost', port=50051, agent_name="Random_V1")
    try:
        client.connect()
    except Exception as e:
        print(f"Error al conectar al servidor: {e}")
        sys.exit(1)
        
    print("\nEstado inicial del tablero:")
    print_board(board)
    
    try:
        # Simulación local (Auto-play entre Negras y Blancas)
        # En una partida real, un lado es el server de Arena, aquí simulamos ambos
        for turn in range(100): 
            is_first_turn = (turn == 0)
            
            # --- Turno de Negras ---
            moves_black = agent_black.get_action(board, is_first_turn=is_first_turn)
            if not moves_black:
                print("Empate: Tablero lleno.")
                break
            
            print(f"Turno {turn + 1}: Negras (X) juegan en {moves_black}")
            board.apply_move(player_id=1, moves=moves_black)
            client.send_move(moves_black) # Simulamos enviar
            print_board(board)
            
            if board.check_victory(player_id=1):
                print("¡Negras (X) han ganado!")
                break
                
            time.sleep(0.5) # Pausa dramática
                
            # --- Turno de Blancas ---
            moves_white = agent_white.get_action(board, is_first_turn=False)
            if not moves_white:
                print("Empate: Tablero lleno.")
                break
                
            print(f"Turno {turn + 1}: Blancas (O) juegan en {moves_white}")
            board.apply_move(player_id=2, moves=moves_white)
            print_board(board)
            
            if board.check_victory(player_id=2):
                print("¡Blancas (O) han ganado!")
                break
                
            time.sleep(0.5)
            
    except KeyboardInterrupt:
        print("\n[!] Partida interrumpida por el usuario.")
    except Exception as e:
        print(f"\n[!] Ocurrió un error inesperado: {e}")
    finally:
        client.disconnect()
        print("Agente terminado limpiamente.")

if __name__ == "__main__":
    main()
