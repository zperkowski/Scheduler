import re


# The format of these data files is:
#     number of problems
#     for each problem in turn:
#        number of jobs (n)
#        for each job i (i=1,...,n) in turn:
#           p(i), a(i), b(i)
def load_data(path):
    n = 0

    results = []
    with open(path) as file:
        problems = int(file.readline())
        for problem in range(problems):
            n = int(file.readline())
            p_list = []
            a_list = []
            b_list = []
            for j in range(n):
                line = file.readline()
                line = re.sub(' +', ' ', line).lstrip()
                p, a, b = line.split(' ')
                p_list.append(int(p))
                a_list.append(int(a))
                b_list.append(int(b))
            result = {
                "problems": problems,
                "n": n,
                "p": p_list,
                "a": a_list,
                "b": b_list
            }
            results.append(result)
    return results
