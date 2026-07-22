"""
Monte Carlo Tree Search (MCTS) para Connect6.

Implementa el algoritmo MCTS descrito en el capítulo 6 de Russell & Norvig,
con las cuatro fases clásicas:

    Selección  -> Expansión -> Simulación (playout) -> Retropropagación

La política de SELECCIÓN es UCB1 (la fórmula UCT de la pág. 209 del libro):

        UCB1(hijo) =  Q(hijo)  +  C * sqrt( ln N(padre) / N(hijo) )

donde:
    Q(hijo) = W(hijo) / N(hijo)   valor promedio (explotación)
    C                              constante de exploración (~sqrt(2))
    N(padre), N(hijo)             número de visitas

La política de SIMULACIÓN (playout) es CONECTABLE (`playout_policy`). Por
defecto usa una heurística de Connect6, pero el proyecto la reemplaza por la
recomendación de una red neuronal profunda entrenada con aprendizaje por
refuerzo (ver ai/dqn_agent.py). Ésa es la combinación MCTS + Deep RL que pide
el enunciado (idea de AlphaGo): en la simulación, cada jugador ya no juega al
azar, sino la mejor jugada que le indica el modelo.

Todo este módulo es Python + numpy puro: se puede ejecutar y verificar SIN
TensorFlow usando la playout policy heurística.
"""

import math
import random

from ai.heuristics import best_heuristic_move, winning_cells

BLACK, WHITE = 1, 2


def _other(player):
    return WHITE if player == BLACK else BLACK


class GameState:
    """
    Estado de una partida de Connect6 pensado para la búsqueda.

    Modela la regla de las fichas: el primer jugador (negras) coloca 1 ficha en
    su primer turno y luego cada jugador coloca 2 fichas por turno. Para el árbol
    trabajamos con jugadas de UNA ficha (plies), llevando la cuenta de cuántas
    fichas le quedan al jugador en el turno actual (`stones_left`). Cuando llegan
    a 0, el turno pasa al rival con 2 fichas.
    """

    def __init__(self, board, to_move, stones_left, winner=None):
        self.board = board            # Connect6Board
        self.to_move = to_move        # jugador que debe mover (1 o 2)
        self.stones_left = stones_left
        self.winner = winner          # None, BLACK, WHITE

    @classmethod
    def initial(cls, board=None):
        """Estado inicial: negras al centro, 1 ficha en el primer turno."""
        from domain.board import Connect6Board
        if board is None:
            board = Connect6Board()
        return cls(board, to_move=BLACK, stones_left=1)

    def legal_moves(self, radius=1):
        return self.board.get_candidate_moves(radius=radius)

    def is_terminal(self):
        return self.winner is not None or self.board.is_full()

    def reward(self):
        """+1 si ganan negras, -1 si ganan blancas, 0 en empate/no terminal."""
        if self.winner == BLACK:
            return 1.0
        if self.winner == WHITE:
            return -1.0
        return 0.0

    def play(self, move):
        """Devuelve un NUEVO estado tras colocar una ficha en `move`."""
        x, y = int(move[0]), int(move[1])
        new_board = self.board.clone()
        won = new_board.wins_at(self.to_move, x, y)
        new_board.grid[x, y] = self.to_move

        winner = self.to_move if won else None
        stones_left = self.stones_left - 1
        to_move = self.to_move
        if winner is None and stones_left == 0:
            to_move = _other(self.to_move)
            stones_left = 2
        return GameState(new_board, to_move, stones_left, winner)


class MCTSNode:
    """Nodo del árbol de búsqueda."""

    __slots__ = ("state", "parent", "move", "player_just_moved",
                 "children", "untried_moves", "N", "W")

    def __init__(self, state, parent=None, move=None, candidates=None):
        self.state = state
        self.parent = parent
        self.move = move  # ficha (x, y) que llevó del padre a este nodo
        # Jugador que hizo la jugada `move` para llegar aquí:
        self.player_just_moved = parent.state.to_move if parent is not None else None
        self.children = {}
        self.untried_moves = list(candidates) if candidates is not None else None
        self.N = 0    # visitas
        self.W = 0.0  # suma de recompensas (desde la óptica de player_just_moved)

    def q_value(self):
        return self.W / self.N if self.N > 0 else 0.0

    def is_fully_expanded(self):
        return self.untried_moves is not None and len(self.untried_moves) == 0

    def ucb1_child(self, c):
        """Selecciona el hijo que maximiza la fórmula UCB1 (UCT)."""
        log_n = math.log(self.N) if self.N > 0 else 0.0
        best, best_score = None, -float("inf")
        for child in self.children.values():
            exploit = child.q_value()                       # Q(hijo)
            explore = c * math.sqrt(log_n / child.N)         # término de exploración
            score = exploit + explore
            if score > best_score:
                best_score, best = score, child
        return best


class MCTS:
    """
    Búsqueda de Monte Carlo. `playout_policy(board, player_id, rng) -> (x, y)`
    decide la jugada durante la simulación. Si es None se usa la heurística.
    """

    def __init__(self, playout_policy=None, n_simulations=200, c=1.4,
                 rollout_depth=50, candidate_radius=1, max_candidates=24,
                 rng=None):
        self.playout_policy = playout_policy
        self.n_simulations = n_simulations
        self.c = c
        self.rollout_depth = rollout_depth
        self.candidate_radius = candidate_radius
        self.max_candidates = max_candidates
        self.rng = rng or random.Random()

    # ---- utilidades ----
    def _candidates(self, state):
        moves = state.legal_moves(radius=self.candidate_radius)
        if self.max_candidates and len(moves) > self.max_candidates:
            # Enfoca la búsqueda en las mejores jugadas según la heurística.
            from ai.heuristics import rank_moves
            moves = rank_moves(state.board, state.to_move, moves,
                               top_k=self.max_candidates)
        return moves

    # ---- 4 fases del MCTS ----
    def search(self, root_state):
        """Ejecuta las simulaciones y devuelve la MEJOR ficha (x, y)."""
        root = MCTSNode(root_state, candidates=self._candidates(root_state))
        if not root.untried_moves:
            return None

        for _ in range(self.n_simulations):
            node = self._select(root)
            if not node.state.is_terminal() and node.untried_moves:
                node = self._expand(node)
            reward = self._simulate(node.state)
            self._backpropagate(node, reward)

        # "Robust child": la jugada más visitada (la más fiable).
        best = max(root.children.values(), key=lambda ch: ch.N)
        return best.move

    def _select(self, node):
        """Baja por el árbol usando UCB1 mientras el nodo esté expandido."""
        while (not node.state.is_terminal()
               and node.untried_moves is not None
               and len(node.untried_moves) == 0
               and node.children):
            node = node.ucb1_child(self.c)
        return node

    def _expand(self, node):
        """Añade un hijo para una jugada no probada."""
        move = node.untried_moves.pop(self.rng.randrange(len(node.untried_moves)))
        child_state = node.state.play(move)
        child = MCTSNode(child_state, parent=node, move=move,
                         candidates=self._candidates(child_state)
                         if not child_state.is_terminal() else [])
        node.children[move] = child
        return child

    def _simulate(self, state):
        """
        Playout: simula la partida hasta un estado terminal (o hasta
        `rollout_depth`) usando la playout policy. Devuelve la recompensa final
        desde la óptica de las negras (+1 gana negras, -1 gana blancas).
        """
        board = state.board.clone()
        to_move = state.to_move
        stones_left = state.stones_left
        winner = state.winner

        depth = 0
        while winner is None and depth < self.rollout_depth:
            if board.is_full():
                break
            move = self._playout_move(board, to_move)
            if move is None:
                break
            x, y = int(move[0]), int(move[1])
            if board.wins_at(to_move, x, y):
                winner = to_move
            board.grid[x, y] = to_move
            stones_left -= 1
            if winner is None and stones_left == 0:
                to_move = _other(to_move)
                stones_left = 2
            depth += 1

        if winner == BLACK:
            return 1.0
        if winner == WHITE:
            return -1.0
        return 0.0

    def _playout_move(self, board, player_id):
        # 1) Si hay jugada ganadora inmediata, tomarla (playout "inteligente").
        wins = winning_cells(board, player_id)
        if wins:
            return wins[0]
        # 2) Bloquear victoria inmediata del rival.
        blocks = winning_cells(board, _other(player_id))
        if blocks:
            return blocks[0]
        # 3) Política de simulación: red neuronal si existe, si no heurística.
        if self.playout_policy is not None:
            move = self.playout_policy(board, player_id, self.rng)
            if move is not None:
                return move
        return best_heuristic_move(board, player_id, self.rng)

    def _backpropagate(self, node, reward):
        """
        Sube la recompensa por el camino. La recompensa está en la óptica de las
        negras; para cada nodo la convertimos a la óptica del jugador que movió
        para llegar a él (por eso el signo depende de `player_just_moved`).
        """
        while node is not None:
            node.N += 1
            if node.player_just_moved == BLACK:
                node.W += reward
            elif node.player_just_moved == WHITE:
                node.W -= reward
            node = node.parent
