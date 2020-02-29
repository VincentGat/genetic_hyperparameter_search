"""This test checks that GeneticSearch is functional.
It also checks that it is usable with a separate scheduler.
"""


import pickle
from pathlib import Path
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import accuracy_score
import numpy as np
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy

SEED = 1324523

a =dict()

try:
    a['allo']
except KeyError:
    print('allo')

with open('log/log3.pkl', 'rb') as file:
    log = pickle.load(file)

log[2]

model = log[2]['best_model']
features = log[2]['features']

def get_real_data():
    csv_path = Path("/home/vincentgatien/Downloads/df_3.csv")
    data = pd.read_csv(csv_path)
    data.drop(columns='Unnamed: 0', inplace=True)
    data = data.iloc[:, :13]
    return data.drop(columns='y_ord').to_numpy(), data.y_ord.to_numpy()


x, y = get_real_data()

rkf = RepeatedStratifiedKFold(n_splits=20, n_repeats=5, random_state=SEED)
data = x[:, features]
perfo = []
model.n_jobs = -1
for train_index, test_index in rkf.split(X=data, y=y):
    x_train_split, x_test_split = data[train_index], data[test_index]
    y_train_split, y_test_split = y[train_index], y[test_index]
    model_split = deepcopy(model)
    model_split.fit(x_train_split, y_train_split)
    y_pred = model_split.predict(x_test_split)
    perfo.append(accuracy_score(y_test_split, y_pred))

np.mean(perfo)
np.std(perfo)
np.min(perfo)
np.max(perfo)

plt.hist(perfo)
plt.show()
plt.clf()

nb_count_list = [i['nb_count'] for i in log]
rf_count_list = [i['rf_count'] for i in log]
catboost_count_list = [i['catboost_count'] for i in log]
lgbm_count_list = [i['lgbm_count'] for i in log]
nb_features_list = [i['nb_features'] for i in log]
min_list = [i['min'] for i in log]
max_list = [i['max'] for i in log]
avg_list = [i['avg'] for i in log]
test_perfo_list = [i['test_perfo'] for i in log]


import matplotlib.pyplot as plt
import seaborn as sns

plt.title('Performance par génération')
plt.xlabel('Génération')
plt.ylabel('Taux de bonnes classifications')
plt.plot(range(51), avg_list)
plt.plot(range(51), max_list)
plt.plot(range(51), test_perfo_list)

plt.legend(['Moyenne', 'Maximum', 'Test'])
plt.savefig('PerformanceGeneration_rd.png')
plt.clf()


plt.title('Nombre de modèles par type de modèles en fonction des générations')
plt.xlabel('Génération')
plt.ylabel('Nombre de modèles')
plt.plot(range(51), nb_count_list)
plt.plot(range(51), rf_count_list)
plt.plot(range(51), catboost_count_list)
plt.plot(range(51), lgbm_count_list)

plt.legend(['nb', 'rf', 'catboost', 'lgbm'])
plt.savefig('nb_model_rd.png')

plt.clf()


plt.title('Nombre de variables du meilleur modèle par génération')
plt.xlabel('Génération')
plt.ylabel('Nombre de variables')
plt.plot(range(51), nb_features_list)

plt.savefig('nb_var_rd.png')
