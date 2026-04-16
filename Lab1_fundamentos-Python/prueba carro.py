from motorTipo import MotorTipo
from carroTipo import CarroTipo
from color import Color
from motor import Motor
from carro import Carro


def main():
    color = Color(120, 180, 255)
    motor = Motor(MotorTipo.HIBRIDO, potencia=200, torque=320, velocidad=220)
    carro = Carro(
        marca="Toyota",
        modelo="Corolla",
        año=2022,
        color=color,
        placa="ABC123",
        tipo=CarroTipo.SEDAN,
        motor=motor,
    )

    print(carro)
    print(motor)
    print(color)
    print("Tipo de carro:", CarroTipo.SEDAN)
    print("Tipo de motor:", MotorTipo.HIBRIDO)

    carro.acelerar()
    carro.parar()
    carro.frenar()
    carro.parar()
    carro.luces()
    carro.luces()


if __name__ == "__main__":
    main()
