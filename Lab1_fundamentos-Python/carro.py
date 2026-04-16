from datetime import datetime

from color import Color
from carroTipo import CarroTipo
from motor import Motor


class Carro:
    def __init__(self, marca, modelo, año, color, placa, tipo, motor):
        self.marca = str(marca)
        self.modelo = str(modelo)
        self.placa = str(placa)
        self.color = color
        self.tipo = tipo
        self.motor = motor
        self.en_movimiento = False
        self.luces_encendidas = False

        current_year = datetime.now().year
        if año <= 1900 or año > current_year:
            print("Error, año inválido")
        self.año = int(año)

    def acelerar(self):
        print("Acelerando el carro")
        self.en_movimiento = True

    def frenar(self):
        if self.en_movimiento:
            print("Frenando el carro")
            self.en_movimiento = False
        else:
            print("El carro está detenido")

    def parar(self):
        if self.en_movimiento:
            print("Parando el carro")
            self.en_movimiento = False
        else:
            print("No hace falta parar el carro, el carro ya está detenido")

    def luces(self):
        if self.luces_encendidas:
            print("Apagando las luces")
            self.luces_encendidas = False
        else:
            print("Prendiendo las luces")
            self.luces_encendidas = True

    def __str__(self):
        return (
            f"Carro(marca={self.marca}, modelo={self.modelo}, año={self.año}, "
            f"color={self.color}, placa={self.placa}, tipo={self.tipo}, motor={self.motor}, "
            f"en_movimiento={self.en_movimiento}, luces_encendidas={self.luces_encendidas})"
        )
