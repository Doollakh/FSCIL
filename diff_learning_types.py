import numpy as np
import matplotlib.pyplot as plt

learnings = []


def load_learning_type(name):
    global learnings
    try:
        learnings.append((name, np.load(f'{name}.npy')))
    except:
        print(f'error loading {name}')


if __name__ == '__main__':
    np.save('joint.npy', np.array([99, 82, 80, 79, 78]))
    np.save('forgetting.npy', np.array([98, 50, 41, 30, 22]))

    x = np.array([20, 25, 30, 35, 40])
    load_learning_type('joint')
    load_learning_type('forgetting')
    colors = ['teal', 'b', 'deeppink', 'tab:red']
    for i, (label, learning) in enumerate(learnings):
        plt.plot(x, learning, colors[i], label=label)
    plt.title('learning')
    plt.xlabel('Classes')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()
