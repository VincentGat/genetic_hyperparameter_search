from numpy import random
import numpy as np
from deap import base, creator, tools
from eaSimple_modif import eaSimple
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.datasets import make_classification
from model import Model
import pandas as pd
from pathlib import Path
from collections import defaultdict

n_features = 45
SEED = 21345
N_JOBS = 1



def get_data():
    """Synthetic binary classification dataset."""
    data, targets = make_classification(
        n_samples=50000,
        n_features=n_features,
        n_informative=12,
        n_redundant=7,
        random_state=134985745,
    )
    return data, targets

def get_real_data():
    csv_path = Path("/home/vincentgatien/Downloads/df_3.csv")
    data = pd.read_csv(csv_path)
    data.drop(columns='Unnamed: 0', inplace=True)
    data = data.iloc[:, :13]
    return data.drop(columns='y_ord').to_numpy(), data.y_ord.to_numpy()


x, y = get_real_data()

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=SEED)

Model.nb_features = x.shape[1]


def mutate(indiv, indpb, features=True, params=True):
    for param_name, param_value in indiv.params.items():
        if random.random() < indpb:
            indiv.params[param_name] = indiv.generate_hyperparams()[param_name]

    indiv.features = np.array([feat if random.random() > indpb else not feat for feat in indiv.features])

    if random.random() < indpb / 2:
        indiv = creator.Individual()

    return indiv,

tested_configurations = {
    'nb': defaultdict(dict),
    'rf': defaultdict(dict),
    'catboost':  defaultdict(dict),
    'lgbm':  defaultdict(dict),
    'tree':  defaultdict(dict),
}
def evaluate(indiv):
    try:
        value = tested_configurations[indiv.model_family][tuple(indiv.params.values())][tuple(indiv.features)]
    except KeyError:
        rkf = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=SEED)
        data = x_train[:, indiv.features]
        perfo = []
        for train_index, test_index in rkf.split(X=data, y=y_train):
            x_train_split, x_test_split = data[train_index], data[test_index]
            y_train_split, y_test_split = y_train[train_index], y_train[test_index]
            model = indiv.model
            model.fit(x_train_split, y_train_split)
            y_pred = model.predict(x_test_split)
            perfo.append(accuracy_score(y_test_split, y_pred))
        value = np.mean(perfo)
        tested_configurations[indiv.model_family][tuple(indiv.params.values())][tuple(indiv.features)] = value
    return value, None


def test_fn(indiv):
    model = indiv.model
    model.fit(x_train[:, indiv.features], y_train)
    return model.score(x_test[:, indiv.features], y_test)


def cxModel(ind1, ind2, swap_indpb=0.5):
    new_feat_ind1 = []
    new_feat_ind2 = []
    for feature1, feature2 in zip(ind1.features, ind2.features):
        if random.random() < swap_indpb:
            new_feat_ind1.append(feature2)
            new_feat_ind2.append(feature1)
        else:
            new_feat_ind1.append(feature1)
            new_feat_ind2.append(feature2)
    ind1.features = np.array(new_feat_ind1)
    ind2.features = np.array(new_feat_ind2)

    if ind1.model_family == ind2.model_family:

        for keys in ind1.params.keys():
            if random.random() < swap_indpb:
                ind1.params[keys], ind2.params[keys] = ind2.params[keys], ind1.params[keys]

    return ind1, ind2


creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", Model, fitness=creator.FitnessMax)

toolbox = base.Toolbox()

toolbox.register("population", tools.initRepeat, list, creator.Individual)

toolbox.register("mate", cxModel)
toolbox.register("mutate", mutate, indpb=0.1)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("evaluate", evaluate)

stats = tools.Statistics(key=lambda ind: ind.fitness.values)
stats.register("avg", np.mean)
stats.register("std", np.std)
stats.register("min", np.min)
stats.register("max", np.max)

pop = toolbox.population(300)
# pool = multiprocessing.Pool(20)
# toolbox.register("map", pool.map)

a = eaSimple(pop, toolbox, cxpb=0.4, mutpb=0.2, ngen=50, test_fn=test_fn, stats=stats, verbose=True)
