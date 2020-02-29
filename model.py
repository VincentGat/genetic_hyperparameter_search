from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import BernoulliNB
from numpy import random
import numpy as np

SEED = 86759
N_JOBS = 1

class Model:
    nb_features = None

    def __init__(self):
        self.model_family = random.choice(['nb', 'rf', 'lgbm', 'catboost', 'tree'], p=[1, 0., 0., 0., 0])
        self._features = random.choice([True, False], self.nb_features)
        self.params = self.generate_hyperparams()

    @property
    def features(self):
        if np.sum(self._features) == 0:
            self._features[random.randint(len(self._features))] = True
        return self._features

    @features.setter
    def features(self, other):
        if np.sum(other) == 0:
            other[random.randint(len(other))] = True
        self._features = other

    @property
    def model(self):
        if self.model_family == 'rf':
            # repair rf if broken
            if self.params['max_features'] > np.sum(self.features):
                self.params['max_features'] = random.randint(1, max(2, np.sum(self.features)))
            if self.params['ccp_alpha'] < 0:
                self.params['ccp_alpha'] = 0
            model = RandomForestClassifier(**self.params, random_state=SEED, n_jobs=N_JOBS)
        elif self.model_family == 'lgbm':
            model = LGBMClassifier(**self.params, random_state=SEED, n_jobs=N_JOBS)
        elif self.model_family == 'nb':
            model = BernoulliNB(**self.params)
        elif self.model_family == 'catboost':
            model = CatBoostClassifier(**self.params, random_state=SEED, verbose=False, thread_count=N_JOBS)
        elif self.model_family == 'tree':
            model = DecisionTreeClassifier(**self.params, max_depth=None, splitter='best', random_state=SEED)
        else:
            model = None
        return model

    def generate_hyperparams(self):
        if self.model_family == 'rf':
            params = self.generate_rf_hyperparams(np.sum(self.features))
        elif self.model_family == 'lgbm':
            params = self.generate_lgbm_hyperparams()
        elif self.model_family == 'nb':
            params = self.generate_nb_hyperparams()
        elif self.model_family == 'catboost':
            params = self.generate_catboost_hyperparams()
        elif self.model_family == 'tree':
            params = self.generate_tree_hyperparams(np.sum(self.features))
        else:
            params = None
        return params

    @staticmethod
    def generate_rf_hyperparams(n_features):
        params = {'n_estimators': random.randint(100, 1000)
            , 'min_samples_split': random.randint(2, 50)
            , 'max_features': random.randint(1, max(2, n_features))
            , 'max_depth': random.randint(80, 105)
            , 'criterion': random.choice(['gini', 'entropy'])
            , 'ccp_alpha': random.uniform(-0.01, 0.025)
                  }
        return params

    @staticmethod
    def generate_lgbm_hyperparams():
        params = {'boosting_type': random.choice(['gbdt', 'dart', 'goss'])
            , 'learning_rate': random.beta(2, 10)
            , 'n_estimators': random.randint(5, 100)
            , 'reg_alpha': random.uniform(0, 0.025)
            , 'reg_lambda': random.uniform(0, 0.025)
            , 'colsample_bytree': random.uniform(0.8, 1)
                  }
        return params

    @staticmethod
    def generate_catboost_hyperparams():
        params = {'iterations': random.randint(5, 100)
            , 'learning_rate': random.beta(2, 10)
            , 'early_stopping_rounds': random.randint(2, 10)
                  }
        return params

    @staticmethod
    def generate_nb_hyperparams():
        params = {} # {'alpha': random.random()}
        return params

    @staticmethod
    def generate_tree_hyperparams(n_features):
        params = {}  #{'max_features': random.randint(1, max(2, n_features))}
        return params