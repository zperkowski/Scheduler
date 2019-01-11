import math
import random

import matplotlib.pyplot as plt
import numpy as np
from keras import utils
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten
from keras.layers import Dense
from keras.losses import categorical_crossentropy
from keras.models import Sequential
from keras.optimizers import Adadelta
from tqdm import tqdm

from generator import generate_set_of_tasks
from loader import load_data, convert_to_numpy_array
from simple_solver import solve


def solve_data(data, h=None, bf_samples=200000):
    if not h:
        all_h = range(20, 90, 20)
        all_h = [h / 100 for h in all_h]
    else:
        all_h = [h]
    scheduled_tasks = []
    possible_orders = []
    for i in range(bf_samples):
        possible_orders.append(random.sample(range(0, len(data[0])), len(data[0])))

    for datum in tqdm(data, desc="Instances"):
        best_order = []
        min_sum_f = 3200000
        for h in all_h:
            for i in range(len(possible_orders)):
                sum_p = sum([p[0] for p in datum])
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
    model.add(Dense(units=32, activation='relu', input_dim=len(train_data[0])))
    model.add(Dense(units=10, activation='softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer='sgd',
                  metrics=['accuracy'])
    train_order = [order[0] for order in train_order]
    train_order = np.asarray(train_order)
    categorized_oder = utils.to_categorical(train_order, num_classes=max(train_order[0]) + 1)
    model.fit(train_data, categorized_oder, epochs=5, batch_size=32)
    print("Training finished")
    classification = model.predict(test_data, batch_size=64)
    return classification


def prepare_conv2d(x_train, y_train, x_test, y_test):
    num_tasks = x_train.shape[1]
    batch_size = 10
    epochs = 10
    input_shape = (num_tasks, 3, 1)
    num_classes = num_tasks
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),
                     activation='relu',
                     input_shape=input_shape))
    model.add(Conv2D(64, (1, 1), activation='relu'))
    model.add(MaxPooling2D(pool_size=(1, 1)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_tasks * num_classes, activation='softmax'))

    model.compile(loss=categorical_crossentropy,
                  optimizer=Adadelta(),
                  metrics=['accuracy'])

    flatten_categorized_y_train = get_flat_categorized_y(y_train, num_classes)
    flatten_categorized_y_test = get_flat_categorized_y(y_test, num_classes)

    history = model.fit(x_train, flatten_categorized_y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        verbose=1,
                        validation_data=(x_test, flatten_categorized_y_test))
    visualize_training(history)
    # score = model.evaluate(x_test, flatten_categorized_y_train, verbose=0)
    # print('Test loss:', score[0])
    # print('Test accuracy:', score[1])
    classification = model.predict(x_test, batch_size=batch_size)
    classification = [c.reshape(num_tasks, num_classes) for c in classification]
    return classification, model


def visualize_training(history):
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()


def get_flat_categorized_y(y_data, num_classes):
    train_order = [order[0] for order in y_data]
    train_order = np.asarray(train_order)
    categorized_y_train = utils.to_categorical(train_order, num_classes=num_classes)
    flatten_categorized_y_train = np.asarray([y.flatten() for y in categorized_y_train])
    return flatten_categorized_y_train


def translate_classification(classification):
    orders = []
    for c in classification:
        order = []
        for task in c:
            value_index = np.argmax(task)
            order.append(value_index)
        orders.append(order)
    return orders


if __name__ == '__main__':
    D = 0.8
    generated_data = generate_set_of_tasks(50000, 10, 1, 20, 1, 10, 1, 15)
    generated_data = convert_to_numpy_array(generated_data)
    order = solve_data(generated_data, D, bf_samples=1000)
    classification, model = prepare_conv2d(generated_data[0:50000],
                                           order[0:50000],
                                           generated_data[45000:100000],
                                           order[45000:100000])
    orders = translate_classification(classification)
    order_train = [t[1] for t in order][5:]

    data = load_data('data/sch10.txt')
    data = convert_to_numpy_array(data)
    order = solve_data(data, D)

    for i in range(len(order)):
        print(str(i) + '\t' + str(order[i][1]))
    print()
    # classification = prepare_model(data[0:5], order[0:5], data[5:9])
    # print(classification)
    # classification = prepare_conv2d(data[0:5], order[0:5], data[5:10], order[5:10])
    classification = model.predict(data, batch_size=10)
    classification = [c.reshape(10, 10) for c in classification]
    orders = translate_classification(classification)
    order_train = [t[1] for t in order][0:10]
    order_score = []
    for i in range(10):
        order_score.append(solve(0.8, data[i], orders[i]))
    for i in range(len(order_score)):
        print(str(i) + '\t' + str(order_score[i]))
