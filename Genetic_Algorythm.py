from numpy import random
import numpy as np
from deap import base, creator, tools, algorithms
import multiprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score

n_features = 45
def get_data():
    """Synthetic binary classification dataset."""
    data, targets = make_classification(
        n_samples=1000,
        n_features=n_features,
        n_informative=12,
        n_redundant=7,
        random_state=134985745,
    )
    return data, targets


x, y = get_data()


def evaluate(individual):
    n_estimators, min_samples_split, max_features, max_depth = individual
    model = RandomForestClassifier(n_estimators=n_estimators
                                   , min_samples_split=min_samples_split, max_features=max_features)
    return cross_val_score(model, X=x, y=y, scoring='neg_log_loss', cv=10).mean(), None


def gen_hyperparam():
    return (random.randint(100, 1000)
            , random.randint(2, 50)
            , random.randint(2, n_features)
            , random.randint(2, 50)
            )


def mutate(individual, indpb):
    individual[0] = random.randint(100, 1000) if random.random() < indpb else individual[0]
    individual[1] = random.randint(2, 50) if random.random() < indpb else individual[1]
    individual[2] = random.randint(2, n_features) if random.random() < indpb else individual[2]
    individual[3] = random.randint(2, 50) if random.random() < indpb else individual[3]
    return individual,


creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()

toolbox.register('hyperparam', gen_hyperparam)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.hyperparam)

toolbox.register("population", tools.initRepeat, list, toolbox.individual)




ind1 = toolbox.individual()
ind2 = toolbox.individual()

tools.cxOnePoint(ind1, ind2)

toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", mutate, indpb=0.2)
toolbox.register("select", tools.selTournament, tournsize=5)
toolbox.register("evaluate", evaluate)

stats = tools.Statistics(key=lambda ind: ind.fitness.values)
stats.register("avg", np.mean)
stats.register("std", np.std)
stats.register("min", np.min)
stats.register("max", np.max)

pop = toolbox.population(100)

pool = multiprocessing.Pool()
toolbox.register("map", pool.map)


a = algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2, ngen=50, stats=stats, verbose=True)
