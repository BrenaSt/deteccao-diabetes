"""Rede Neural (Perceptron Multicamadas, MLP).

Média de 5 redes pequenas (sementes diferentes) treinadas com o solver
lbfgs, que otimiza com todos os dados de uma vez e dá resultados estáveis
em bases pequenas. Além das colunas do config.py, recebe as colunas de
corte clínico (HB_65, HB_57, GLIC_126, N_EXAMES_AUSENTES).
"""
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.neural_network import MLPClassifier

from src import config
from src.evaluation import executar
from src.preprocessing import CORTES_CLINICOS

NOME = "neural_network"
OPCOES_PREP = {"numericas": config.COLS_NUMERICAS + CORTES_CLINICOS}


class MLPMedia(ClassifierMixin, BaseEstimator):
    """Treina uma MLP por semente e tira a média das probabilidades.
    Responde diabetes quando a probabilidade média passa do `limiar`."""

    def __init__(self, hidden_layer_sizes=(4,), alpha=10.0, activation="relu",
                 sementes=(42, 101, 202, 303, 404), limiar=0.5):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.alpha = alpha
        self.activation = activation
        self.sementes = sementes
        self.limiar = limiar

    def fit(self, X, y):
        base = MLPClassifier(hidden_layer_sizes=self.hidden_layer_sizes, alpha=self.alpha,
                             activation=self.activation, solver="lbfgs",
                             max_iter=4000, max_fun=50000)
        self.redes_ = [clone(base).set_params(random_state=s).fit(X, y) for s in self.sementes]
        self.classes_ = self.redes_[0].classes_
        return self

    def predict_proba(self, X):
        return np.mean([r.predict_proba(X) for r in self.redes_], axis=0)

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] > self.limiar).astype(int)


def criar_modelo():
    return MLPMedia()


GRADE = {
    "modelo__hidden_layer_sizes": [(2,)],  # neurônios na camada escondida
    "modelo__alpha": [3.0],                # regularização (maior = rede mais simples)
    "modelo__activation": ["tanh"],
    "modelo__limiar": [0.55],              # probabilidade mínima para responder diabetes
}


def main(aninhada: bool = False, sufixo: str = ""):
    return executar(NOME + sufixo, criar_modelo(), GRADE, aninhada=aninhada,
                    opcoes_prep=OPCOES_PREP)


if __name__ == "__main__":
    main()
