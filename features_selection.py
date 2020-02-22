from numpy import random
import numpy as np
from deap import base, creator, tools
from algo_modif import eaSimple
import multiprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score, train_test_split
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from pathlib import Path
import pandas as pd



seed = 12234
def get_real_data():
    csv_path = Path("/home/vincentgatien/Downloads/df_3.csv")
    data = pd.read_csv(csv_path)
    data.drop(columns='Unnamed: 0', inplace=True)
    return data.drop(columns='y_ord').to_numpy(), data.y_ord.to_numpy()

def get_data():
    """Synthetic binary classification dataset."""
    data, targets = make_classification(
        n_samples=10000,
        n_features=45,
        n_informative=12,
        n_redundant=7,
        random_state=seed,
    )
    return data, targets


x, y = get_real_data()
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)


def configure_model(individual):
    if individual['model_family'] == 'rf':
        model = RandomForestClassifier(**individual['params'], random_state=seed)
    elif individual['model_family'] == 'lgbm':
        model = LGBMClassifier(**individual['params'], random_state=seed, n_jobs=1)
    elif individual['model_family'] == 'nb':
        model = BernoulliNB(**individual['params'])
    elif individual['model_family'] == 'catboost':
        model = CatBoostClassifier(**individual['params'], random_state=seed, verbose=False, thread_count=1)
    else:
        model = None
    return model


def evaluate(individual):
    features = individual['feature_selection']
    model = configure_model(individual)
    return cross_val_score(model, X=x_train[:, features], y=y_train, scoring='f1', cv=10).mean(), None


def test_fn(individual):
    features = individual['feature_selection']
    model = configure_model(individual)
    model.fit(x_train[:, features], y_train)
    return model.score(x_test[:, features], y_test)


def generate_rf_hyperparams(n_features):
    params = {'n_estimators': random.randint(100, 1000)
        , 'min_samples_split': random.randint(2, 50)
        , 'max_features': random.randint(2, n_features)
        , 'max_depth': random.randint(2, 50)
              }
    return params


def generate_lgbm_hyperparams():
    params = {'boosting_type': random.choice(['gbdt', 'dart', 'goss'])
        , 'learning_rate': random.beta(2, 10)
        , 'n_estimators': random.randint(5, 100)
              }
    return params


def generate_catboost_hyperparams():
    params = {'iterations': random.randint(5, 100)
        , 'learning_rate': random.beta(2, 10)
        , 'early_stopping_rounds': random.randint(2, 10)
    }
    return params


def generate_nb_hyperparams():
    params = {'alpha': random.random()}
    return params


def generate_hyperparams(model_family, n_features):
    if model_family == 'rf':
        params = generate_rf_hyperparams(n_features)
    elif model_family == 'lgbm':
        params = generate_lgbm_hyperparams()
    elif model_family == 'nb':
        params = generate_nb_hyperparams()
    elif model_family == 'catboost':
        params = generate_catboost_hyperparams()
    else:
        params = {}
    return params


def generate_individual():
    model_family = random.choice(['lgbm', 'rf', 'nb', 'catboost'], p=[0.25, 0.5, 0, 0.25])
    features = random.choice([True, False], x.shape[1])
    params = generate_hyperparams(model_family, np.sum(features))
    genes = {'feature_selection': features
        , 'params': params
        , 'model_family': model_family
             }
    return genes


def mutate_rf_param(individual, indpb):
    params = individual['params']
    params['n_estimators'] = random.randint(100, 1000) if random.random() < indpb else params['n_estimators']
    params['min_samples_split'] = random.randint(2, 50) if random.random() < indpb else params[
        'min_samples_split']
    params['max_features'] = random.randint(2, np.sum(individual['feature_selection'])) if random.random() < indpb else \
        min(params['max_features'], np.sum(individual['feature_selection']))
    params['max_depth'] = random.randint(2, 50) if random.random() < indpb else params['max_depth']

    individual = repairRF(individual)
    return individual


def mutate_lgbm_param(individual, indpb):
    params = individual['params']

    params['boosting_type'] = random.choice(['gbdt', 'dart', 'goss']) if random.random() < indpb else params[
        'boosting_type']
    params['learning_rate'] = random.beta(2, 10) if random.random() < indpb else params['learning_rate']
    params['n_estimators'] = random.randint(5, 100) if random.random() < indpb else params['n_estimators']
    return individual

def mutate_nb_param(individual, indpb):
    params = individual['params']
    params['alpha'] = random.random() if random.random() < indpb else params['alpha']
    return individual

def mutate_catboost_param(individual, indpb):
    params = individual['params']

    params['iterations'] = random.randint(5, 100) if random.random() < indpb else params['iterations']
    params['learning_rate'] = random.beta(2, 10) if random.random() < indpb else params['learning_rate']
    params['early_stopping_rounds'] = random.randint(2, 10) if random.random() < indpb else params['early_stopping_rounds']
    return individual

def mutate(individual, indpb):
    features = individual['feature_selection']
    for idx, feat in enumerate(features):
        if random.random() < indpb:
            features[idx] = not feat

    if individual['model_family'] == 'rf':
        individual = mutate_rf_param(individual, indpb)

    elif individual['model_family'] == 'lgbm':
        individual = mutate_lgbm_param(individual, indpb)

    elif individual['model_family'] == 'nb':
        individual = mutate_nb_param(individual, indpb)

    elif individual['model_family'] == 'catboost':
        individual = mutate_catboost_param(individual, indpb)

    return individual,


def repairRF(individual):
    individual['params']['max_features'] = min(individual['params']['max_features'],
                                               np.sum(individual['feature_selection']))
    return individual



def cxModel(ind1, ind2, swap_indpb=0.5):
    features_ind1 = ind1['feature_selection']
    features_ind2 = ind2['feature_selection']
    for idx in range(len(features_ind1)):
        if random.random() < swap_indpb:
            features_ind1[idx], features_ind2[idx] = features_ind2[idx], features_ind1[idx]
    if ind1['model_family'] == ind2['model_family']:
        params_ind1 = ind1['params']
        params_ind2 = ind2['params']
        for keys in params_ind1.keys():
            if random.random() < swap_indpb:
                params_ind1[keys], params_ind2[keys] = params_ind2[keys], params_ind1[keys]
        if ind1['model_family'] == 'rf':
            ind1 = repairRF(ind1)
            ind2 = repairRF(ind2)
    return ind1, ind2


creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", dict, fitness=creator.FitnessMax)

toolbox = base.Toolbox()

toolbox.register("individual", tools.initIterate, creator.Individual, generate_individual)

toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("mate", cxModel)
toolbox.register("mutate", mutate, indpb=0.4)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("evaluate", evaluate)

stats = tools.Statistics(key=lambda ind: ind.fitness.values)
stats.register("avg", np.mean)
stats.register("std", np.std)
stats.register("min", np.min)
stats.register("max", np.max)

pop = toolbox.population(300)

pool = multiprocessing.Pool()
toolbox.register("map", pool.map)

a = eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2, ngen=50, test_fn=test_fn, stats=stats, verbose=True)

import matplotlib.pyplot as plt

plt.hist(random.beta(2, 10, 10000))
plt.show()
b = tools.selBest(a[0], 1)[0]
sum(b['feature_selection'])
a[0]