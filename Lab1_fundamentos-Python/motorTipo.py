from enum import Enum


class MotorTipo(Enum):
    COMBUSTION = 1
    ELECTRICO = 2
    HIBRIDO = 3

    def __str__(self):
        return self.name
