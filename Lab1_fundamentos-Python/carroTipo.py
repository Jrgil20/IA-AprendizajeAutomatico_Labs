from enum import Enum


class CarroTipo(Enum):
    SUV = 1
    CROSSOVER = 2
    DEPORTIVO = 3
    SEDAN = 4
    PICKUP = 5
    STATIOWAGON = 6
    MINIVAN = 7
    VAN = 8
    COUPE = 9
    HATCHBACK = 10

    def __str__(self):
        return self.name
