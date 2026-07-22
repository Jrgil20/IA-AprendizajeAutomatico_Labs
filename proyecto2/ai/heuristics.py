"""
Heurísticas propias del juego Connect6.

Estas funciones implementan las "estrategias propias del juego" que pide el
enunciado: detectar situaciones borde (ganar en una jugada, bloquear la victoria
del rival) y puntuar jugadas por su valor táctico. Se usan en dos lugares:

  1. En el agente (MCTSAgent), como pre-chequeo antes de lanzar el MCTS:
     si hay una jugada ganadora se toma directamente; si el rival amenaza con
     ganar, se bloquea.
  2. Como "playout policy" de respaldo del MCTS cuando todavía no hay una red
     neuronal entrenada (permite probar el MCTS sin TensorFlow).
"""

# Direcciones (horizontal, vertical, diagonal \, diagonal /)
_DIRECTIONS = [(1, 0), (0, 1), (1, 1), (1, -1)]


def winning_cells(board, player_id, candidates=None):
    """
    Devuelve las casillas vacías donde `player_id` GANA de inmediato al colocar
    una ficha (completa 6 en línea). Si se le pasa `player_id` del rival, sirve
    para detectar las casillas que hay que bloquear.
    """
    if candidates is None:
        candidates = board.get_candidate_moves(radius=1)
    wins = []
    for (x, y) in candidates:
        if board.grid[x, y] == 0 and board.wins_at(player_id, x, y):
            wins.append((x, y))
    return wins


def _line_score(board, player_id, x, y):
    """
    Puntúa una casilla vacía para `player_id` según cuántas fichas propias y
    del rival hay alineadas a su alrededor. Premia extender líneas propias y
    (con algo menos de peso) cortar líneas del rival.
    """
    n = board.size
    opponent = 2 if player_id == 1 else 1
    score = 0.0

    for dx, dy in _DIRECTIONS:
        own = 0
        opp = 0
        # Ventana de 5 casillas a cada lado a lo largo de la dirección.
        for sign in (1, -1):
            for step in range(1, 6):
                i, j = x + sign * dx * step, y + sign * dy * step
                if not (0 <= i < n and 0 <= j < n):
                    break
                v = board.grid[i, j]
                if v == player_id:
                    # Fichas propias más cercanas valen más.
                    own += (6 - step)
                elif v == opponent:
                    opp += (6 - step)
                    break  # una ficha rival corta nuestra línea en esa dirección
                else:
                    break  # casilla vacía: dejamos de contar consecutivas
        score += own * 1.0 + opp * 0.8  # atacar y defender

    # Sesgo suave hacia el centro del tablero.
    c = n / 2.0
    score += 1.0 - (abs(x - c) + abs(y - c)) / (2 * n)
    return score


def rank_moves(board, player_id, candidates=None, top_k=None):
    """
    Ordena las jugadas candidatas de mejor a peor según la heurística.
    Antepone cualquier jugada ganadora inmediata y luego los bloqueos.
    Devuelve una lista de tuplas (x, y).
    """
    if candidates is None:
        candidates = board.get_candidate_moves(radius=1)

    opponent = 2 if player_id == 1 else 1
    my_wins = set(winning_cells(board, player_id, candidates))
    blocks = set(winning_cells(board, opponent, candidates))

    def priority(move):
        x, y = move
        base = _line_score(board, player_id, x, y)
        if move in my_wins:
            base += 1e6  # ganar ya
        elif move in blocks:
            base += 1e5  # impedir que el rival gane
        return base

    ranked = sorted(candidates, key=priority, reverse=True)
    if top_k is not None:
        ranked = ranked[:top_k]
    return ranked


def best_heuristic_move(board, player_id, rng=None):
    """
    Devuelve la mejor jugada según la heurística. Se usa como playout policy de
    respaldo (sin red neuronal). `rng` opcional para desempatar con algo de azar
    entre las mejores, dando variedad a las simulaciones del MCTS.
    """
    ranked = rank_moves(board, player_id)
    if not ranked:
        return None
    if rng is None:
        return ranked[0]
    # Elige entre las 3 mejores para no ser totalmente determinista.
    k = min(3, len(ranked))
    return ranked[rng.randrange(k)]
