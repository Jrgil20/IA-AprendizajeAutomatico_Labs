"""
Punto de entrada para jugar en la plataforma Connect6 Arena.

Conecta el agente principal (MCTS + red neuronal + heurísticas) al servidor de
Arena por gRPC. Si existe un modelo entrenado (dqn_model.keras) se usa como
política de simulación del MCTS; si no, el MCTS juega con su playout heurístico.

PASOS PREVIOS (una sola vez):
    pip install grpcio grpcio-tools            # requiere Python 3.9 - 3.12
    python -m grpc_tools.protoc -I proto --python_out=. --grpc_python_out=. proto/connect6.proto

USO:
    python run_arena.py --team MiEquipo --address localhost:50051
    # (la dirección también puede venir en la variable de entorno SERVER_ADDR)
"""

import argparse
import os

from ai.mcts_agent import MCTSAgent
from network.client import Connect6ArenaClient


def make_agent_factory(n_simulations):
    """Devuelve una función factory(player_id) -> agente, para el color que asigne Arena."""
    def factory(player_id):
        for model_file in ("dqn_model.keras", "dqn_model.h5"):
            if os.path.exists(model_file):
                try:
                    from ai.dqn_agent import DQNAgent  # import perezoso (necesita TensorFlow)
                    dqn = DQNAgent(player_id=player_id, epsilon=0.0)
                    dqn.load(model_file)
                    print(f"[Arena] Modelo '{model_file}' cargado como playout del MCTS.")
                    return MCTSAgent(player_id=player_id, dqn_agent=dqn,
                                     n_simulations=n_simulations)
                except Exception as e:
                    print(f"[Arena] No se pudo cargar '{model_file}' "
                          f"({type(e).__name__}); se usará el playout heurístico.")
                    break
        return MCTSAgent(player_id=player_id, dqn_agent=None, n_simulations=n_simulations)
    return factory


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Agente Connect6 para Connect6 Arena.")
    parser.add_argument("--team", default="MCTS_DeepRL", help="Nombre del equipo.")
    parser.add_argument("--address", default=None,
                        help="host:puerto del servidor (o variable SERVER_ADDR).")
    parser.add_argument("--sims", type=int, default=120,
                        help="Simulaciones de MCTS por jugada.")
    args = parser.parse_args()

    client = Connect6ArenaClient(
        agent_factory=make_agent_factory(args.sims),
        team_name=args.team,
        address=args.address,
    )
    client.play()
