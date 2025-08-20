"""Math helper utilities."""

import math


def distance(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])
