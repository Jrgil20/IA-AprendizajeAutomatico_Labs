"""
Agente de juego de Connect6 basado en MCTS + red neuronal + heurísticas.

Es el agente principal que pide el enunciado. Para cada turno decide 1 ficha
(primer turno de negras) o 2 fichas con la siguiente lógica:

    1. Apertura: si el tablero está vacío, juega al centro (9, 9).
    2. Victoria inmediata: si puede completar 6 en línea, lo hace.
    3. Bloqueo: si el rival amenaza con ganar en su próxima ficha, lo bloquea.
    4. En otro caso, ejecuta MCTS (selección UCB1) para elegir la ficha; durante
       la simulación del MCTS se usa la red neuronal como playout policy (o la
       heurística si no se ha entrenado ninguna red).

El método `get_action` devuelve la lista de fichas del turno, en el mismo
formato que espera main.py y el cliente de Connect6 Arena.
"""

import random

from ai.mcts import MCTS, GameState
from ai.heuristics import winning_cells

BLACK, WHITE = 1, 2


class MCTSAgent:
    def __init__(self, player_id, dqn_agent=None, n_simulations=200, c=1.4,
                 rollout_depth=50, max_candidates=24, use_heuristics=True,
                 rng=None):
        self.player_id = player_id
        self.opponent = WHITE if player_id == BLACK else BLACK
        self.dqn_agent = dqn_agent
        self.use_heuristics = use_heuristics
        self.rng = rng or random.Random()

        playout_policy = dqn_agent.policy if dqn_agent is not None else None
        self.mcts = MCTS(
            playout_policy=playout_policy,
            n_simulations=n_simulations,
            c=c,
            rollout_depth=rollout_depth,
            max_candidates=max_candidates,
            rng=self.rng,
        )

    def _choose_stone(self, board, stones_left):
        """Elige UNA ficha aplicando heurísticas y, si hace falta, MCTS."""
        if self.use_heuristics:
            # 2. Victoria inmediata propia.
            my_wins = winning_cells(board, self.player_id)
            if my_wins:
                return my_wins[0]
            # 3. Bloqueo de victoria inmediata del rival.
            opp_wins = winning_cells(board, self.opponent)
            if opp_wins:
                return opp_wins[0]

        # 4. Búsqueda MCTS.
        state = GameState(board.clone(), to_move=self.player_id,
                          stones_left=stones_left)
        move = self.mcts.search(state)
        if move is None:  # tablero sin candidatos (borde): cae a cualquier válida
            valid = board.get_valid_moves()
            move = self.rng.choice(valid) if valid else None
        return move

    def get_action(self, board, is_first_turn=False):
        """Devuelve la lista de fichas (1 o 2) a jugar este turno."""
        num_stones = 1 if is_first_turn else 2

        # 1. Apertura al centro.
        if is_first_turn and int((board.grid != 0).sum()) == 0:
            c = board.size // 2
            return [(c, c)]

        moves = []
        work = board.clone()  # tablero de trabajo para decidir las 2 fichas
        for k in range(num_stones):
            if work.is_full():
                break
            stone = self._choose_stone(work, stones_left=num_stones - k)
            if stone is None:
                break
            stone = (int(stone[0]), int(stone[1]))
            moves.append(stone)
            work.grid[stone[0], stone[1]] = self.player_id
            # Si esta ficha ya gana, no hace falta la segunda.
            if work.wins_at(self.player_id, stone[0], stone[1]):
                break
        return moves
