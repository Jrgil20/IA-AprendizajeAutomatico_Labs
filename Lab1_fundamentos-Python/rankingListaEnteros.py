entrada = input("Ingrese los números enteros separados por espacios: ")

numeros = []
for parte in entrada.split():
    numeros.append(int(parte))

ordenados = []
indice = 0
while indice < len(numeros):
    actual = numeros[indice]
    ya_en_ordenados = False
    j = 0
    while j < len(ordenados):
        if ordenados[j] == actual:
            ya_en_ordenados = True
            break
        j += 1

    if not ya_en_ordenados:
        menor = actual
        k = 0
        while k < len(numeros):
            ya_en_ordenados = False
            m = 0
            while m < len(ordenados):
                if ordenados[m] == numeros[k]:
                    ya_en_ordenados = True
                    break
                m += 1
            if not ya_en_ordenados and numeros[k] < menor:
                menor = numeros[k]
            k += 1
        ordenados.append(menor)

    indice += 1

rankings = []
indice = 0
while indice < len(numeros):
    valor = numeros[indice]
    posicion = 1
    j = 0
    while j < len(ordenados):
        if ordenados[j] == valor:
            break
        posicion += 1
        j += 1
    rankings.append(posicion)
    indice += 1

print(rankings)
