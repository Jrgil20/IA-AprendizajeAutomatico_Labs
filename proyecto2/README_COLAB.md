# Guía de Entrenamiento en Google Colab (Agente DQN)

Debido al tamaño del tablero de Connect6 (19x19), entrenar una red neuronal profunda (DQN) desde cero requiere un alto poder de cómputo. Se recomienda encarecidamente utilizar las GPUs gratuitas de Google Colab para acelerar masivamente este proceso.

A continuación, se presentan dos métodos para lograrlo.

---

## Método 1: A través de GitHub (Recomendado)

Si tienes este proyecto subido a un repositorio de GitHub público, sigue estos pasos:

### 1. Preparar el Entorno en Colab
1. Ve a [Google Colab](https://colab.research.google.com/) y crea un **Nuevo Cuaderno**.
2. En el menú superior, ve a **Entorno de ejecución** -> **Cambiar tipo de entorno de ejecución**.
3. En "Acelerador por hardware", selecciona **T4 GPU** y guarda.

### 2. Clonar y Configurar
En la primera celda del cuaderno, clona tu repositorio ejecutando:
```python
!git clone https://github.com/tu-usuario/tu-repositorio.git
```

En la segunda celda, entra a la carpeta del proyecto e instala las dependencias:
```python
%cd tu-repositorio/proyecto2
!pip install -r requirements.txt
```

### 3. Entrenar el Modelo
*(Opcional: Abre el archivo `train.py` haciendo doble clic desde la barra lateral de archivos de Colab y cambia `episodes=50` a `episodes=10000`).*

En una tercera celda, inicia el entrenamiento intensivo:
```python
!python train.py
```

### 4. Descargar el Modelo
Una vez que la consola muestre que el entrenamiento finalizó, busca el archivo `dqn_model.h5` en la barra lateral izquierda de Archivos (dentro de tu carpeta clonada). Dale clic derecho, selecciona **Descargar**, y ubícalo en la carpeta `proyecto2` de tu PC local. Al ejecutar tu `main.py` local, la IA usará los nuevos conocimientos.

---

## Método 2: Subida Directa (Método Express)

Si no utilizas Git/GitHub o tu repositorio es muy privado:

1. **Comprime localmente** toda la carpeta `proyecto2` de tu PC en un archivo llamado `proyecto2.zip`. (Te sugiero subir los episodios a 10,000 en el código de `train.py` antes de comprimir).
2. Abre Colab, crea un nuevo cuaderno y activa la **GPU T4** (como en el Paso 1 del Método anterior).
3. Abre el menú de **Archivos** en la barra lateral izquierda de Colab (el ícono de carpeta).
4. **Arrastra y suelta** tu archivo `proyecto2.zip` en ese panel para subirlo temporalmente a la nube.
5. Ejecuta las siguientes celdas de código en orden para arrancar:

**Celda 1 (Descomprimir):**
```python
!unzip proyecto2.zip
```

**Celda 2 (Moverse al directorio e instalar dependencias):**
```python
%cd proyecto2
!pip install -r requirements.txt
```

**Celda 3 (Arrancar entrenamiento):**
```python
!python train.py
```

Al terminar, solo descarga el archivo `dqn_model.h5` de la barra lateral hacia tu PC local como se explicó previamente.
