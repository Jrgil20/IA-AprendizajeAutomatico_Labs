anio = int(input("Ingrese un año: "))

if anio < 1900 or anio > 2200:
	print("Error, año inválido")
else:
	if anio % 4 == 0:
		if anio % 100 == 0:
			if anio % 400 == 0:
				print("El año es bisiesto")
			else:
				print("El año no es bisiesto")
		else:
			print("El año es bisiesto")
	else:
		print("El año no es bisiesto")
