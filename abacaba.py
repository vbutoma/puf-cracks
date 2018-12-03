import random
import math
import numpy as np
import pickle as pkl
import time
from deap import creator, base, tools, algorithms, benchmarks, cma
from nn import build_logistic_model
from scipy.stats import norm
from functools import partial
import math
from math import factorial as fact

np.random.seed(42)

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("evaluate", benchmarks.rastrigin)


def sigmoid(x):
    x = max(x, -10)
    x = min(x, 10)
    return 1/(1 + math.exp(-x))


def bits_dist(mu=128, sigma=127):
    return partial(norm.pdf, loc=mu, scale=sigma)


def init_bits_dist():
    def C(n, m):
        p = fact(m)
        q = fact(n) * fact(m - n)
        return p // q

    p = [C(i, 255) / 2 ** 255 for i in range(256)]
    m_x = 0
    d_x = 0
    for i in range(256):
        m_x += p[i] * i
    for i in range(256):
        d_x += p[i] * (i - m_x) ** 2
    return bits_dist(mu=m_x, sigma=math.sqrt(d_x))


bits_distribution = init_bits_dist()
bits_multiplier = 1 / bits_distribution(127.5)


def loss(x):
    fitted, mask = delta_list(x_val, x)
    count = len(fitted)
    good = 0
    # start = time.time()
    cnt = [0, 0]
    for i in range(count):
        cnt[fitted[i]] += 1
        if fitted[i] == y_val[i]:
            good += 1
    # print(cnt, good, abs(np.corrcoef(fitted, y_val[:count])[0][1]))
    # print(time.time() - start, good, count, good/count)
    # res = np.corrcoef(fitted, y_val[:count])
    # print(sum(mask), bits_multiplier * bits_distribution(sum(mask)))
    return (good/count * bits_multiplier * bits_distribution(sum(mask)), )
    # return (np.abs(res[0][1]), )


N = 8


def load_dumped(n, method=3):
    start_time = time.time()
    with open("dumped3/train_data_{}_{}.pkl".format(method, n), 'rb') as f:
        train_data = pkl.load(f)
    with open("dumped3/val_data_{}_{}.pkl".format(method, n), 'rb') as f:
        val_data = pkl.load(f)
    with open("dumped3/test_data_{}_{}.pkl".format(method, n), 'rb') as f:
        test_data = pkl.load(f)
    print('data loaded in {}'.format(time.time() - start_time))
    return train_data, val_data, test_data


train_data, val_data, test_data = load_dumped(n=8)
x_train, y_train = list(map(np.array, zip(*train_data)))
x_val, y_val = list(map(np.array, zip(*test_data)))
x_test, y_test = list(map(np.array, zip(*test_data)))
x_val = x_val[:1000]
y_val = y_val[:1000]

model = build_logistic_model(input_dim=8, output_dim=2)
model.load_weights('logit_weights.h5')


def delta_list(a, b):
    n = len(a)
    mask = [0 if sigmoid(_) <= 0.5 else 1 for _ in b]
    challenges = [[] for _ in range(n)]
    # get 1st ones
    for i in range(len(mask)):
        if mask[i] and len(challenges[0]) < N:
            for j in range(n):
                challenges[j].append(a[j][i])
    row = [[] for _ in range(n)]
    cur = [challenges[i][0] for i in range(n)]
    for i in range(n):
        row[i].append(cur[i] if cur[i] else -1)
    for i in range(1, N):
        for j in range(n):
            cur[j] ^= challenges[j][i]
            row[j].append(cur[j] if cur[j] else -1)
    fitted = model.predict_classes(np.asarray(row))
    return fitted, mask


def generate_challenges(a, b):
    n = len(a)
    mask = [0 if sigmoid(_) <= 0.5 else 1 for _ in b]
    challenges = [[] for _ in range(n)]
    # get 1st ones
    for i in range(len(mask)):
        if mask[i] and len(challenges[0]) < N:
            for j in range(n):
                challenges[j].append(a[j][i])
    row = [[] for _ in range(n)]
    cur = [challenges[i][0] for i in range(n)]
    for i in range(n):
        row[i].append(cur[i] if cur[i] else -1)
    for i in range(1, N):
        for j in range(n):
            cur[j] ^= challenges[j][i]
            row[j].append(cur[j] if cur[j] else -1)
    return row, mask

def class_eq():
    cnt = [0, 0]
    for _ in y_val:
        cnt[int(_)] += 1
    print('Zeros: {}, Ones: {}, {}'.format(cnt[0], cnt[1], cnt[0]/cnt[1]))


CHECKPOINT_FREQ = 10

if __name__ == "__main__":
    class_eq()

    bit_cnt = 255
    gen_size = bit_cnt
    strategy = cma.Strategy(centroid=[0.0] * gen_size, sigma=1.0, lambda_=20)
    toolbox.register("generate", strategy.generate, creator.Individual)
    toolbox.register("evaluate", loss)
    toolbox.register("update", strategy.update)

    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])
    ngen = 4100
    start_gen = 0
    if start_gen:
        with open('checkpoints/cp_{}.pkl'.format(start_gen), "rb") as cp_file:
            cp = pkl.load(cp_file)
            population = cp["population"]
            toolbox.update(population)
            start_gen = cp["generation"]
            hof = cp["halloffame"]
            logbook = cp["logbook"]
            random.setstate(cp["rndstate"])

    for gen in range(start_gen, ngen):
        # Generate a new population
        population = toolbox.generate()
        for p in population:
            for i in range(len(p)):
                p[i] = max(p[i], -10)
                p[i] = min(p[i], 10)
        # Evaluate the individuals
        fitnesses = toolbox.map(toolbox.evaluate, population)
        for ind, fit in zip(population, fitnesses):
            ind.fitness.values = fit
        hof.update(population)

        # Update the strategy with the evaluated individuals
        toolbox.update(population)
        record = stats.compile(population) if stats is not None else {}
        logbook.record(gen=gen, nevals=len(population), **record)
        print(logbook.stream)
        if gen % CHECKPOINT_FREQ == 0:
            # Fill the dictionary using the dict(key=value[, ...]) constructor
            cp = dict(population=population, generation=gen, halloffame=hof,
                      logbook=logbook, rndstate=random.getstate())
            with open("checkpoints/cp_{}.pkl".format(gen), "wb") as cp_file:
                pkl.dump(cp, cp_file)
    population = toolbox.generate()[0]
    print(population)