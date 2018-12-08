import math
from loader import load_data
from simple_solver import solve

results = load_data('data/sch10.txt')
for result in results:
    for h in range(20, 90, 20):
        h /= 100
        sum_p = sum(result.get("p"))
        d = math.floor(sum_p * h)
        print("h: " + str(h) + " d: " + str(d))
        print(solve(d, result))
    print()
