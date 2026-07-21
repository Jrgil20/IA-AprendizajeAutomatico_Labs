import numpy as np

class Connect6Board:
    def __init__(self):
        # 0: Vacío, 1: Negras (Juega primero, 1 ficha en el primer turno), 2: Blancas
        self.size = 19
        self.grid = np.zeros((self.size, self.size), dtype=np.int8)
        self.turn_count = 0

    def get_valid_moves(self):
        """Retorna una lista de tuplas (x, y) con las casillas vacías."""
        return list(zip(*np.where(self.grid == 0)))

    def is_full(self):
        """Verifica si el tablero está lleno."""
        return len(self.get_valid_moves()) == 0

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
