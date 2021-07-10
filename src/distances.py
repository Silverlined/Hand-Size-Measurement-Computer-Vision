from numpy import array, linalg

def getSlopeIntercept(point1, point2):
    """Compute slope and intercept of line"""

    x1, y1 = point1
    x2, y2 = point2
    a = (y2 - y1) / (x2 - x1)
    b = y1 - a * x1

    return a, b

def lineAt(a, b, x):
    """Compute y of line at x"""

    return a * x + b

def vertDistFingers(tip1, tip2, a, b):
    """Compute vertical distance between fingers"""

    dist1 = tip1[1] - lineAt(a, b, tip1[0])
    dist2 = tip2[1] - lineAt(a, b, tip2[0])

    return abs(dist1 - dist2)

def horDistFingers(tip1, tip2, a, b):
    """Compute horizontal distance between fingers"""

    line_point1 = array([tip1[0], lineAt(a, b, tip1[0])])
    line_point2 = array([tip2[0], lineAt(a, b, tip2[0])])

    dist = linalg.norm(line_point1-line_point2)
    return dist
