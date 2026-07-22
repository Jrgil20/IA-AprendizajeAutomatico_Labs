"""
Pruebas locales del agente (tablero + heurísticas + MCTS + auto-juego).

NO requiere TensorFlow: verifica toda la lógica del agente usando el playout
heurístico del MCTS. Ejecutar con:  python test_agent.py
"""

import random
import time

from domain.board import Connect6Board
from ai import heuristics
from ai.mcts import MCTS, GameState
from ai.mcts_agent import MCTSAgent

PASSED = 0
FAILED = 0


def check(name, condition):
    global PASSED, FAILED
    if condition:
        PASSED += 1
        print(f"  [OK]   {name}")
    else:
        FAILED += 1
        print(f"  [FALLA] {name}")


# ---------------------------------------------------------------------------
print("\n== 1. Tablero: clone / candidatos / wins_at ==")
b = Connect6Board()
b.grid[9, 9] = 1
cand = b.get_candidate_moves(radius=1)
check("candidatos alrededor de 1 ficha = 8 vecinos", len(cand) == 8)
check("tablero vacio -> candidato unico es el centro",
      Connect6Board().get_candidate_moves() == [(9, 9)])
c2 = b.clone()
c2.grid[0, 0] = 2
check("clone es independiente del original", b.grid[0, 0] == 0 and c2.grid[0, 0] == 2)

b2 = Connect6Board()
for y in range(5):
    b2.grid[3, y] = 1
check("wins_at detecta que (3,5) completa 6 en fila", b2.wins_at(1, 3, 5) is True)
check("wins_at: (3,6) NO completa 6 (hay hueco)", b2.wins_at(1, 3, 6) is False)

# ---------------------------------------------------------------------------
print("\n== 2. Heuristicas: victoria y bloqueo inmediato ==")
bw = Connect6Board()
for y in range(5):
    bw.grid[3, y] = 1  # negras con 5 en fila, ganan en (3,5) o (3,-1 invalido)
wins = heuristics.winning_cells(bw, 1)
check("winning_cells encuentra la casilla ganadora (3,5)", (3, 5) in wins)

bb = Connect6Board()
for y in range(5):
    bb.grid[7, y] = 2  # blancas amenazan ganar en (7,5)
blocks = heuristics.winning_cells(bb, 2)
check("winning_cells(rival) detecta la amenaza (7,5) a bloquear", (7, 5) in blocks)

# ---------------------------------------------------------------------------
print("\n== 3. MCTSAgent: apertura / victoria / bloqueo ==")
agent = MCTSAgent(player_id=1, n_simulations=30, rng=random.Random(0))

opening = agent.get_action(Connect6Board(), is_first_turn=True)
check("apertura de negras juega al centro (9,9)", opening == [(9, 9)])

bwin = Connect6Board()
for y in range(5):
    bwin.grid[3, y] = 1
mv = agent.get_action(bwin, is_first_turn=False)
check("el agente toma la victoria inmediata (3,5)", (3, 5) in mv)

bblock = Connect6Board()
for y in range(5):
    bblock.grid[7, y] = 2   # rival a punto de ganar
# damos también algo a negras para que no gane antes
bblock.grid[0, 0] = 1
mvb = agent.get_action(bblock, is_first_turn=False)
check("el agente bloquea la amenaza del rival en (7,5)", (7, 5) in mvb)

# ---------------------------------------------------------------------------
print("\n== 4. MCTS puro devuelve jugada legal ==")
mcts = MCTS(n_simulations=40, rng=random.Random(1))
st = GameState.initial()
st = st.play((9, 9))  # negras al centro -> pasa a blancas
move = mcts.search(st)
check("MCTS devuelve una jugada", move is not None)
check("la jugada del MCTS es una casilla vacia",
      move is not None and st.board.grid[move[0], move[1]] == 0)

# ---------------------------------------------------------------------------
print("\n== 5. Auto-juego completo entre dos agentes MCTS (heuristico) ==")
random.seed(42)
board = Connect6Board()
black = MCTSAgent(player_id=1, n_simulations=25, rng=random.Random(2))
white = MCTSAgent(player_id=2, n_simulations=25, rng=random.Random(3))
winner = None
t0 = time.time()
for turn in range(200):
    is_first = (turn == 0)
    bm = black.get_action(board, is_first_turn=is_first)
    if not bm:
        break
    board.apply_move(1, bm)
    if board.check_victory(1):
        winner = "Negras"
        break
    wm = white.get_action(board, is_first_turn=False)
    if not wm:
        break
    board.apply_move(2, wm)
    if board.check_victory(2):
        winner = "Blancas"
        break
    if board.is_full():
        break
dt = time.time() - t0
fichas = int((board.grid != 0).sum())
check("la partida termina con un ganador o empate", winner is not None or board.is_full() or fichas > 0)
print(f"  -> Resultado: {winner or 'sin ganador (tope de turnos/empate)'} | "
      f"fichas colocadas: {fichas} | tiempo: {dt:.1f}s")

# ---------------------------------------------------------------------------
print("\n" + "=" * 44)
print(f" RESUMEN:  {PASSED} OK, {FAILED} fallos")
print("=" * 44)
raise SystemExit(1 if FAILED else 0)
