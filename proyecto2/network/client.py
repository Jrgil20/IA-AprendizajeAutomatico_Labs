class Connect6Client:
    """
    Esqueleto del cliente gRPC para Connect6.
    Por ahora es un Dummy Client que no requiere gRPC real,
    listo para reemplazarse cuando se tenga el .proto.
    """
    def __init__(self, host='localhost', port=50051, agent_name="Agent"):
        self.host = host
        self.port = port
        self.agent_name = agent_name
        self.connected = False

    def connect(self):
        print(f"[Network] Simulando conexión gRPC a {self.host}:{self.port} como '{self.agent_name}'...")
        self.connected = True
        print("[Network] Conexión simulada exitosa.")

    def disconnect(self):
        if self.connected:
            print("[Network] Desconectando del servidor de forma limpia.")
            self.connected = False

    def send_move(self, moves):
        """Envía el movimiento al servidor."""
        print(f"[Network] Simulando envío de movimiento al servidor gRPC: {moves}")

    def wait_for_turn(self):
        """Espera a que sea el turno del jugador y retorna el estado del tablero (simulado)."""
        pass
