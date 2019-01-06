
def solve(d, result, order=None):
    if not order:
        order = range(len(result[0]))
    scheduled_tasks = []
    current_time = 0
    sum_f = 0
    for i in order:
        if current_time < d and current_time + result[i][0][0] <= d:
            f = result[i][0][0] * result[i][1][0]
        elif current_time >= d and current_time + result[i][0][0] > d:
            f = result[i][0][0] * result[i][2][0]
        else:
            f = ((d - current_time) * result[i][1][0]) + (result[i][0][0] - (d - current_time)) * result[i][2][0]

        # task = {
        #     "id": i,
        #     "start_time": current_time,
        #     "f": f
        # }
        # scheduled_tasks.append(task)

        current_time += result[i][0][0]
        sum_f += f
    return sum_f
