ajedrez = input("Integrantes del club de ajedrez: ").split()
futbol = input("Integrantes del club de fútbol: ").split()
natacion = input("Integrantes del club de natación: ").split()
lectura = input("Integrantes del club de lectura: ").split()

comunes = []
for nombre in ajedrez:
    if nombre in futbol and nombre in natacion and nombre in lectura:
        if nombre not in comunes:
            comunes.append(nombre)

if comunes:
    print(";".join(comunes))
else:
    print("No hay nombres en común")
