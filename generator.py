import random


def generate_set_of_tasks(num_of_problems, n, min_p, max_p, min_a, max_a, min_b, max_b):
    problems = []
    for problem in range(num_of_problems):
        result = {
            "problems": num_of_problems,
            "n": n,
            "p": [random.randint(min_p, max_p) for _ in range(n)],
            "a": [random.randint(min_a, max_a) for _ in range(n)],
            "b": [random.randint(min_b, max_b) for _ in range(n)]
        }
        problems.append(result)
    return problems
