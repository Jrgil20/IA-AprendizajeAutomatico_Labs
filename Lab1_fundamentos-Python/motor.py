from motorTipo import MotorTipo


class Motor:
    def __init__(self, tipo, potencia, torque, velocidad):
        if not isinstance(tipo, MotorTipo):
            raise ValueError("tipo debe ser un MotorTipo")
        self.tipo = tipo
        self.potencia = int(potencia)
        self.torque = int(torque)
        self.velocidad = int(velocidad)

    def __str__(self):
        return (
            f"Motor(tipo={self.tipo}, potencia={self.potencia} HP, "
            f"torque={self.torque} Nm, velocidad={self.velocidad} km/h)"
        )
