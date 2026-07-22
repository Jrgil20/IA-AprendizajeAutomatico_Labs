"""
Clientes de red para el agente de Connect6.

  * Connect6Client       : cliente simulado (dummy) usado por main.py para la
                           demostración local, sin necesidad de gRPC.
  * Connect6ArenaClient  : cliente gRPC REAL para la plataforma Connect6 Arena
                           (servicio GameServer, ver proto/connect6.proto).

Para usar el cliente real primero hay que generar los stubs de Python:
    pip install grpcio grpcio-tools           # Python 3.9 - 3.12
    python -m grpc_tools.protoc -I proto --python_out=. --grpc_python_out=. proto/connect6.proto
Esto crea connect6_pb2.py y connect6_pb2_grpc.py en la carpeta del proyecto.
Luego se arranca con run_arena.py.
"""

import os
import queue

from domain.board import Connect6Board

BLACK, WHITE = 1, 2


# ---------------------------------------------------------------------------
# Cliente simulado (para la demo local de main.py)
# ---------------------------------------------------------------------------
class Connect6Client:
    """Esqueleto que imita la conexión sin usar gRPC (demo local)."""

    def __init__(self, host='localhost', port=50051, agent_name="Agent"):
        self.host = host
        self.port = port
        self.agent_name = agent_name
        self.connected = False

    def connect(self):
        print(f"[Network] (simulado) conectando a {self.host}:{self.port} como '{self.agent_name}'...")
        self.connected = True

    def disconnect(self):
        if self.connected:
            print("[Network] (simulado) desconectando.")
            self.connected = False

    def send_move(self, moves):
        print(f"[Network] (simulado) enviando movimiento: {moves}")

    def wait_for_turn(self):
        pass


# ---------------------------------------------------------------------------
# Cliente gRPC real para Connect6 Arena
# ---------------------------------------------------------------------------
class Connect6ArenaClient:
    """
    Cliente del servicio `GameServer` de Connect6 Arena.

    El protocolo es un stream bidireccional `Play`:
      - el cliente envía mensajes PlayerAction (register_team / move / resign);
      - el servidor envía mensajes GameState (tablero, turno, color, resultado).

    Se le pasa `agent_factory(player_id) -> agente`, donde el agente expone
    `get_action(board, is_first_turn) -> [(x, y), ...]`. El color propio lo
    indica el servidor (my_color), por eso el agente se crea al conocerlo.
    """

    def __init__(self, agent_factory, team_name="MCTS_DeepRL", address=None):
        self.agent_factory = agent_factory
        self.team_name = team_name
        # El servidor de Arena suele indicar la dirección por variable de entorno.
        self.address = address or os.environ.get("SERVER_ADDR", "localhost:50051")
        self._agent = None
        self._board = Connect6Board()  # tablero persistente sincronizado con el servidor

    # ---- utilidades de conversión con el proto ----
    @staticmethod
    def _import_grpc():
        try:
            import grpc
            import connect6_pb2 as pb
            import connect6_pb2_grpc as pb_grpc
            return grpc, pb, pb_grpc
        except ImportError as e:
            raise RuntimeError(
                "No se encontraron gRPC o los stubs. Instala y genera con:\n"
                "  pip install grpcio grpcio-tools\n"
                "  python -m grpc_tools.protoc -I proto --python_out=. "
                "--grpc_python_out=. proto/connect6.proto"
            ) from e

    def _sync_board(self, state):
        """Reconstruye el tablero persistente a partir del GameState del servidor."""
        # 1) Si el servidor manda el tablero completo, es la fuente de verdad.
        if state.board and len(state.board) == self._board.size:
            for i, row in enumerate(state.board):
                for j, cell in enumerate(row.cells):
                    self._board.grid[i, j] = int(cell)  # PlayerColor == nuestro id
            return
        # 2) Si no, aplicamos las últimas fichas del oponente al tablero persistente.
        opp = WHITE if state.my_color == BLACK else BLACK
        for p in state.opponent_stones:
            x, y = int(p.x), int(p.y)
            if self._board.grid[x, y] == 0:
                self._board.grid[x, y] = opp

    def _ensure_stone_count(self, moves, required, player_id):
        """Garantiza que la jugada tenga exactamente `required` fichas."""
        moves = [(int(x), int(y)) for (x, y) in moves][:required]
        if len(moves) < required:
            from ai.heuristics import rank_moves
            occupied = set(moves)
            for cand in rank_moves(self._board, player_id):
                c = (int(cand[0]), int(cand[1]))
                if c not in occupied and self._board.grid[c[0], c[1]] == 0:
                    moves.append(c)
                    occupied.add(c)
                    if len(moves) == required:
                        break
        return moves

    def play(self):
        grpc, pb, pb_grpc = self._import_grpc()
        print(f"[Arena] Conectando a {self.address} como equipo '{self.team_name}'...")
        channel = grpc.insecure_channel(self.address)
        stub = pb_grpc.GameServerStub(channel)

        send_q = queue.Queue()

        def request_gen():
            # Lo primero: registrar el equipo.
            yield pb.PlayerAction(register_team=self.team_name)
            while True:
                action = send_q.get()
                if action is None:
                    return
                yield action

        try:
            for state in stub.Play(request_gen()):
                if state.message:
                    print(f"[Arena] {state.message}")

                if state.status == pb.GameState.FINISHED:
                    print(f"[Arena] Partida finalizada. Resultado: "
                          f"{pb.GameResult.Name(state.result)} | "
                          f"ganador: {pb.PlayerColor.Name(state.winner)}")
                    break

                if not state.is_my_turn:
                    continue

                # Es nuestro turno: sincronizar tablero y decidir jugada.
                self._sync_board(state)
                player_id = int(state.my_color)
                if self._agent is None:
                    self._agent = self.agent_factory(player_id)
                    print(f"[Arena] Jugamos con color {pb.PlayerColor.Name(state.my_color)}.")

                required = int(state.stones_required) or 2
                is_first = (required == 1)
                moves = self._agent.get_action(self._board, is_first_turn=is_first)
                moves = self._ensure_stone_count(moves, required, player_id)

                # Aplicar en el tablero persistente y enviar.
                for (x, y) in moves:
                    self._board.grid[x, y] = player_id
                print(f"[Arena] Jugamos {moves}")
                send_q.put(pb.PlayerAction(
                    move=pb.Move(stones=[pb.Point(x=x, y=y) for (x, y) in moves])))
        finally:
            send_q.put(None)
            channel.close()
            print("[Arena] Conexión cerrada.")
