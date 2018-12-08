
def solve(d, result):
    scheduled_tasks = []
    current_time = 0
    sum_f = 0
    for i in range(result['n']):
        if current_time < d and current_time + result['p'][i] <= d:
            f = result['p'][i] * result['a'][i]
        elif current_time >= d and current_time + result['p'][i] > d:
            f = result['p'][i] * result['b'][i]
        else:
            f = ((d - current_time) * result['a'][i]) + (result['p'][i] - (d - current_time)) * result['b'][i]

        task = {
            "start_time": current_time,
            "f": f
        }
        scheduled_tasks.append(task)
        current_time += result['p'][i]
        sum_f += task['f']
    return scheduled_tasks, sum_f
