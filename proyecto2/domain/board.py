import numpy as np

class Connect6Board:
    def __init__(self):
        # 0: Vacío, 1: Negras (Juega primero, 1 ficha en el primer turno), 2: Blancas
        self.size = 19
        self.grid = np.zeros((self.size, self.size), dtype=np.int8)
        self.turn_count = 0

    def clone(self):
        """Copia profunda y barata del tablero (para la simulación del MCTS)."""
        new_board = Connect6Board()
        new_board.grid = self.grid.copy()
        new_board.turn_count = self.turn_count
        return new_board

    def get_valid_moves(self):
        """Retorna una lista de tuplas (x, y) con las casillas vacías."""
        return list(zip(*np.where(self.grid == 0)))

    def get_candidate_moves(self, radius=1):
        """
        Poda de relevancia (imprescindible en tableros 19x19): sólo considera
        casillas vacías que estén a `radius` de alguna ficha ya colocada.
        Si el tablero está vacío, devuelve únicamente el centro.
        Reduce el factor de ramificación de 361 a unas pocas decenas.
        """
        occupied = np.argwhere(self.grid != 0)
        if len(occupied) == 0:
            c = self.size // 2
            return [(c, c)]

        candidates = set()
        n = self.size
        for (ox, oy) in occupied:
            for dx in range(-radius, radius + 1):
                for dy in range(-radius, radius + 1):
                    x, y = int(ox) + dx, int(oy) + dy
                    if 0 <= x < n and 0 <= y < n and self.grid[x, y] == 0:
                        candidates.add((x, y))
        return list(candidates)

    def is_full(self):
        """Verifica si el tablero está lleno."""
        return len(self.get_valid_moves()) == 0

    def wins_at(self, player_id, x, y):
        """
        Chequeo RÁPIDO: ¿colocar `player_id` en (x, y) forma una línea de >=6?
        Sólo examina las 4 direcciones que pasan por (x, y), en vez de escanear
        todo el tablero. Se usa para detectar victoria/bloqueo inmediato y como
        chequeo de terminalidad durante las simulaciones del MCTS.
        """
        x, y = int(x), int(y)
        if self.grid[x, y] != 0 and self.grid[x, y] != player_id:
            return False
        n = self.size
        directions = [(1, 0), (0, 1), (1, 1), (1, -1)]
        for dx, dy in directions:
            count = 1  # la ficha hipotética en (x, y)
            # hacia adelante
            i, j = x + dx, y + dy
            while 0 <= i < n and 0 <= j < n and self.grid[i, j] == player_id:
                count += 1
                i += dx
                j += dy
            # hacia atrás
            i, j = x - dx, y - dy
            while 0 <= i < n and 0 <= j < n and self.grid[i, j] == player_id:
                count += 1
                i -= dx
                j -= dy
            if count >= 6:
                return True
        return False

    def get_state(self, player_id):
        """
        Retorna el estado del tablero desde la perspectiva del jugador.
        1.0 para sus fichas, -1.0 para las del oponente, 0.0 vacío.
        Con forma (19, 19, 1) para Keras CNN.
        """
        state = np.zeros((self.size, self.size, 1), dtype=np.float32)
        state[self.grid == player_id, 0] = 1.0
        state[(self.grid != 0) & (self.grid != player_id), 0] = -1.0
        return state

    def apply_move(self, player_id, moves):
        """
        Aplica un movimiento. Recibe el ID del jugador y una lista de coordenadas.
        """
        for x, y in moves:
            # Casteamos a int nativo en caso de que numpy devuelva numpy.int64
            x, y = int(x), int(y)
            if self.grid[x, y] == 0:
                self.grid[x, y] = player_id
            else:
                raise ValueError(f"Movimiento inválido: la casilla ({x},{y}) está ocupada.")
        self.turn_count += 1

    def step(self, player_id, move):
        """
        Aplica un movimiento unitario (x, y) como en Gym.
        Retorna: (reward, done, is_valid)
        """
        x, y = move
        if self.grid[x, y] != 0:
            return -10.0, False, False # Recompensa altamente negativa por movimiento inválido
            
        self.grid[x, y] = player_id
        
        if self.check_victory(player_id):
            return 1.0, True, True  # Victoria
            
        if self.is_full():
            return 0.0, True, True  # Empate
            
        return 0.0, False, True     # Continua el juego

    def check_victory(self, player_id):
        """
        Verifica si hay 6 (o más) fichas alineadas horizontal, vertical o diagonalmente
        para el jugador especificado.
        """
        # Comprobar filas
        for i in range(self.size):
            row = self.grid[i, :]
            if self._check_line(row, player_id):
                return True
            
        # Comprobar columnas
        for j in range(self.size):
            col = self.grid[:, j]
            if self._check_line(col, player_id):
                return True
                
        # Comprobar diagonales (principal e inversa)
        # En una matriz de 19x19, las diagonales válidas van de offset -13 a 13
        # porque necesitamos al menos 6 elementos.
        for offset in range(-(self.size - 6), self.size - 5):
            diag1 = np.diagonal(self.grid, offset=offset)
            if self._check_line(diag1, player_id):
                return True
            
            diag2 = np.diagonal(np.fliplr(self.grid), offset=offset)
            if self._check_line(diag2, player_id):
                return True

        return False

    def _check_line(self, line, player_id):
        """Busca si hay al menos 6 fichas consecutivas de `player_id` en un vector 1D."""
        count = 0
        for val in line:
            if val == player_id:
                count += 1
                if count >= 6:
                    return True
            else:
                count = 0
        return False
