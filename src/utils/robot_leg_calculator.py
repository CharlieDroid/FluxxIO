import math
import numpy as np
from engineering_notation import EngNumber as eng


def rad2deg(rad):
    return rad * 180 / math.pi


def deg2rad(deg):
    return deg * math.pi / 180


def getLK(_L1, _L2, _thK):
    return math.sqrt(_L2**2 + _L1**2 - 2*_L2*_L1*math.cos(_thK))


def getthS(_LK, _L1, _L2, _L3):
    th1 = math.acos((_LK**2 + _L1**2 - _L2**2)/(2*_LK*_L1))
    th2 = math.acos((_LK**2 + _L3**2 - _L1**2)/(2*_LK*_L3))
    return math.pi - th1 - th2


def getLhr(_L1, _L4, _thS, _r):
    _L = math.sqrt(_L1**2 + _L4**2 - 2*_L1*_L4*math.cos(_thS))
    _hr = _L + _r
    return _L, _hr


if __name__ == "__main__":
    L1 = 120e-3
    L2 = L1 * 1/2.2
    L3 = L1 * 1/3.6
    L4 = L1 * (1 + 1/18)
    r = 50e-3 * 0.5
    print(f"L1={eng(L1)}m\tL2={eng(L2)}m\tL3={eng(L3)}m\tL4={eng(L4)}m\tr={eng(r)}m")

    print("\nFinding viable angles:")
    minimum = float("inf")
    maximum = float("-inf")
    max_L = 0
    min_L = 0
    for thK in np.arange(-45 + 45, 90 + 45, 0.01):
        try:
            LK = getLK(L1, L2, deg2rad(thK))
            thS = getthS(LK, L1, L2, L3)
            L, hr = getLhr(L1, L4, thS, r)
            if minimum == float("inf"):
                if thK < minimum:
                    minimum = thK
                    min_L = L
            else:
                if thK > maximum:
                    maximum = thK
                    max_L = L
        except Exception as e:
            pass

    print(f"Minimum viable angle: {eng(minimum)}deg")
    print(f"Maximum viable angle: {eng(maximum)}deg")
    print(f"Angle Difference: {eng(maximum - minimum)}deg")
    print(f"Minimum viable length: {eng(min_L)}m")
    print(f"Maximum viable length: {eng(max_L)}m")
    print(f"Length Difference: {eng(max_L - min_L)}m")

    print("\nResting Position (Half of DOF):")
    thK = ((maximum - minimum) / 2) + minimum
    print(f"thK={eng(thK)}deg")
    print(f"limits: +-{eng((maximum - minimum) / 2)}deg")
    thK = deg2rad(thK)

    LK = getLK(L1, L2, thK)
    print(f"LK={eng(LK)}m")

    thS = getthS(LK, L1, L2, L3)
    print(f"thS={eng(rad2deg(thS))}deg")

    L, hr = getLhr(L1, L4, thS, r)
    print(f"L={eng(L)}m\thr={eng(hr)}m")
