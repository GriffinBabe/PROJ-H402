import matplotlib.pyplot as plt
import csv
import sys

if __name__ == '__main__':
    if len(sys.argv) != 2:
        raise Exception('Program arguments: python plot_log.py <csv_file>')

    csv_path = sys.argv[1]
    accuracy = []
    loss = []
    with open(csv_path, 'r', newline='') as csv_file:
        data_reader = csv.reader(csv_file, delimiter=',')
        data_reader.__next__()
        for row in data_reader:
            accuracy.append(float(row[1]))
            loss.append(float(row[2]))

    plt.style.use('ggplot')

    fig, ax = plt.subplots()

    ax.set_xlabel('epochs')
    ax.set_ylabel('accuracy')
    a, = ax.plot(accuracy, 'r', label='accuracy')

    ax2 = ax.twinx()

    ax2.set_ylabel('loss (categorical crossentropy)')
    b, = ax2.plot(loss, 'b', label='loss')

    p = [a, b]
    ax.legend(p, [p_.get_label() for p_ in p], loc='upper left')

    plt.show()

