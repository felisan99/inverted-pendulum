import numpy as np

def seno(t: float, frec: int, a: int) -> float:
    return a * np.sin(frec * np.pi * t)
