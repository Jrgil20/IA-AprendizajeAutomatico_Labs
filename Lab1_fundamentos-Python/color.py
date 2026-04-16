class Color:
    def __init__(self, red, green, blue):
        if not (1 <= red <= 255 and 1 <= green <= 255 and 1 <= blue <= 255):
            raise ValueError("Los valores de color deben estar entre 1 y 255")
        self.red = red
        self.green = green
        self.blue = blue

    def __str__(self):
        return f"Color({self.red}, {self.green}, {self.blue})"
