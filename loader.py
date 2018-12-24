import re
import numpy as np


# The format of these data files is:
#     number of problems
#     for each problem in turn:
#        number of jobs (n)
#        for each job i (i=1,...,n) in turn:
#           p(i), a(i), b(i)
def load_data(path):
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


def convert_to_numpy_array(data):
    arrays = []
    for datum in data:
        values = [datum['p'], datum['a'], datum['b']]
        array = np.array(values, dtype=int)
        arrays.append(array)
    return arrays


def save_data(path, result, scheduled_task):
    len = result['n']
    with open(path, 'w') as file:
        file.write(result["sum_f"])
        file.write(str(len))
        for i in range(len):
            t_id = scheduled_task[i]["id"]
            file.write(result['p'][t_id] + "\t" + result['a'][t_id] + "\t" + result['b'][t_id])
