import math
from loader import load_data

results = load_data('data/sch10.txt')
for result in results:
    h = 0.2
    d = math.floor(sum(result.get("p")) * h)
    print("h: " + str(h) + " d: " + str(d))
    h = 0.4
    d = math.floor(sum(result.get("p")) * h)
    print("h: " + str(h) + " d: " + str(d))
    h = 0.6
    d = math.floor(sum(result.get("p")) * h)
    print("h: " + str(h) + " d: " + str(d))
    h = 0.8
    d = math.floor(sum(result.get("p")) * h)
    print("h: " + str(h) + " d: " + str(d))
    print()
