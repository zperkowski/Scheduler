import math
import itertools

from keras.utils import to_categorical

from loader import load_data, save_data, convert_to_numpy_array
from simple_solver import solve
from keras.models import Sequential
from keras.layers import Dense
from tqdm import tqdm


def solve_data(data, h=None):
    if not h:
        all_h = range(20, 90, 20)
        all_h = [h / 100 for h in all_h]
    else:
        all_h = [h]
    scheduled_tasks = []
    for datum in tqdm(data, desc="Instances"):
        best_order = []
        min_sum_f = 3200000
        for h in all_h:
            possible_orders = list(itertools.permutations([i for i in range(len(datum[0]))]))
            possible_orders = possible_orders[:10000]
            for i in tqdm(range(len(possible_orders)), desc="Order permutations", mininterval=0.5):
                sum_p = sum(datum[0])
                d = math.floor(sum_p * h)
                sum_f = solve(d, datum, possible_orders[i])
                if sum_f < min_sum_f:
                    best_order = possible_orders[i]
                    min_sum_f = sum_f
                # save_data('data/sch10_output.txt', datum, scheduled_tasks)
        scheduled_tasks.append((best_order, min_sum_f))
    return scheduled_tasks


def prepare_model(train_data, train_order, test_data):
    model = Sequential()
    model.add(Dense(units=64, activation='relu', input_shape=train_data[0].shape))
    model.add(Dense(units=10, activation='softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer='sgd',
                  metrics=['accuracy'])
    model.fit(train_data, train_order, epochs=5, batch_size=32)
    classification = model.predict(test_data, batch_size=128)
    return classification


if __name__ == '__main__':
    data = load_data('data/sch10.txt')
    data = convert_to_numpy_array(data)
    order = solve_data(data, 0.8)
    print(order)
    classification = prepare_model(data[0:5], order[0:5], data[5:9])
    print(classification)
