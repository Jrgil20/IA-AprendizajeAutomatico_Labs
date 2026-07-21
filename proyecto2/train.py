import sys
import numpy as np
from domain.board import Connect6Board
from ai.dqn_agent import DQNAgent
from ai.random_agent import RandomAgent

def train_dqn(episodes=50, batch_size=32, target_update=5):
    print("="*40)
    print(" Iniciando Entrenamiento del Agente DQN")
    print("="*40)
    
    # DQN jugará con Negras (ID=1), el aleatorio con Blancas (ID=2)
    agent_dqn = DQNAgent(player_id=1, epsilon=1.0, epsilon_min=0.1, epsilon_decay=0.99)
    agent_random = RandomAgent(player_id=2)
    
    for episode in range(episodes):
        board = Connect6Board()
        done = False
        turn = 0
        total_loss = 0
        total_reward = 0
        
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
                
                # Interacción con el Entorno (MDP)
                reward, done, is_valid = board.step(player_id=1, move=move)
                next_state = board.get_state(player_id=1)
                
                # Guardar experiencia (S, A, R, S', Done)
                agent_dqn.remember(state, action_flat, reward, next_state, done)
                total_reward += reward
                
                # Entrenar un paso de los pesos
                loss = agent_dqn.train_step(batch_size)
                total_loss += loss
                
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
                reward_w, done, _ = board.step(player_id=2, move=w_move)
                if done:
                    # Si Blancas gana, DQN recibe castigo
                    # Almacenamos la experiencia perdedora desde la perspectiva de Negras
                    next_state = board.get_state(player_id=1)
                    # Tomamos la última acción que hicimos (action_flat de arriba) y le asignamos la penalización
                    agent_dqn.remember(state, action_flat, -1.0, next_state, done)
                    break
                    
            turn += 1
            
        # Actualizar la red objetivo periódicamente
        if episode % target_update == 0:
            agent_dqn.update_target_network()
            
        print(f"Episodio {episode+1}/{episodes} | Recompensa Total: {total_reward} | Epsilon: {agent_dqn.epsilon:.3f}")
        
    print("Guardando modelo en 'dqn_model.h5'...")
    agent_dqn.save("dqn_model.h5")
    print("Entrenamiento Completado. Ya puedes usar main.py")

if __name__ == "__main__":
    # Ajustar para más episodios en la vida real (ej. 10,000)
    train_dqn(episodes=50)
