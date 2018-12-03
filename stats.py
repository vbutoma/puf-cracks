from matplotlib import pyplot as plt


def tree():
    x = [8 * (i + 1) for i in range(16)]
    y_tree = [0.21941, 0.11609, 0.12634, 0.04012, 0.08753, 0.078, 0.09722, 0.05999,
              0.06753, 0.05125, 0.05446, 0.06771, 0.07345, 0.05983, 0.06916, 0.079]
    plt.plot(x, y_tree)
    plt.xticks(x)
    plt.xlabel('N')
    plt.ylabel('Binary Error')
    plt.savefig('binary_error.png')
    plt.close()


def nn():
    x = [8 * (i + 1) for i in range(16)]
    y_tree = [0.59368, 0.88247, 0.87065, 0.96279, 0.91577, 0.92261, 0.90961, 0.94729,
              0.94019, 0.97518, 0.98646, 0.94415, 0.42976, 0.47089, 0.95893, 0.61692]
    plt.plot(x, y_tree)
    plt.xticks(x)
    plt.xlabel('N')
    plt.ylabel('Accuracy')
    plt.savefig('nn_acc.png')
    plt.close()


def logit():
    x = [8 * (i + 1) for i in range(16)]
    y_tree = [0.78142, 0.88215, 0.87262, 0.96328, 0.91573, 0.92521, 0.90457,
              0.94773, 0.94127, 0.97261, 0.97925, 0.9466, 0.94843, 0.97876, 0.96394, 0.98391]
    plt.plot(x, y_tree)
    plt.xticks(x)
    plt.xlabel('N')
    plt.ylabel('Accuracy')
    plt.savefig('logit_acc.png')
    plt.close()


def gen():
    x = [8 * (i + 1) for i in range(16)]
    y_tree = [0.6, 0.61, 0.65, 0.66, 0.68, 0.67, 0.67,
              0.67, 0.68, 0.64, 0.67, 0.6678, 0.689, 0.69, 0.65, 0.699]
    y_tree1 = [0.67, 0.69, 0.7, 0.68, 0.69, 0.75, 0.7,
              0.73, 0.76, 0.75, 0.75, 0.73, 0.8, 0.79, 0.89, 0.84]
    plt.plot(x, y_tree)
    plt.plot(x, y_tree1)
    plt.xticks(x)
    plt.xlabel('N')
    plt.ylabel('Accuracy')
    plt.savefig('Gen1.png')
    plt.close()


if __name__ == "__main__":
    gen()



