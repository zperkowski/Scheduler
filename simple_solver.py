
def solve(d, result, order=None):
    if not order:
        order = range(len(result[0]))
    scheduled_tasks = []
    current_time = 0
    sum_f = 0
    for i in order:
        if current_time < d and current_time + result[0][i] <= d:
            f = result[0][i] * result[1][i]
        elif current_time >= d and current_time + result[0][i] > d:
            f = result[0][i] * result[2][i]
        else:
            f = ((d - current_time) * result[1][i]) + (result[0][i] - (d - current_time)) * result[2][i]

        # task = {
        #     "id": i,
        #     "start_time": current_time,
        #     "f": f
        # }
        # scheduled_tasks.append(task)

        current_time += result[0][i]
        sum_f += f
    return sum_f
