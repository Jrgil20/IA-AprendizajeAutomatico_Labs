secuencia = input("Ingrese la secuencia de nucleótidos: ").strip()
longitud = int(input("Ingrese el tamaño a explorar: "))

conteo = {}
indice = 0
while indice + longitud <= len(secuencia):
    sub = secuencia[indice:indice + longitud]
    if sub in conteo:
        conteo[sub] += 1
    else:
        conteo[sub] = 1
    indice += 1

items = list(conteo.items())
items.sort(key=lambda x: x[1])

for sub, frecuencia in items:
    print(sub, frecuencia)
