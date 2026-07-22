import random

class RandomAgent:
    def __init__(self, player_id):
        self.player_id = player_id

    def get_action(self, board, is_first_turn=False):
        """
        Calcula el siguiente movimiento.
        Retorna 1 coordenada si es el primer turno del juego, o 2 coordenadas para el resto.
        """
        valid_moves = board.get_valid_moves()
        num_stones = 1 if is_first_turn else 2
        
        if len(valid_moves) < num_stones:
            # Retorna lo que quede si no hay suficiente espacio para la cantidad solicitada
            chosen = valid_moves
        else:
            chosen = random.sample(valid_moves, num_stones)

        # Casteo a int nativo (get_valid_moves devuelve numpy.int64).
        return [(int(x), int(y)) for x, y in chosen]
