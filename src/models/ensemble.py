"""Ensemble por votação: Random Forest + Rede Neural + KNN.

Cada membro recebe o CSV bruto, faz o próprio pré-processamento e devolve
a probabilidade de diabetes; o ensemble tira a média ponderada.
Os membros são cópias (sem treino) dos modelos salvos em artifacts/, com
os mesmos hiperparâmetros. Por isso, os modelos individuais precisam ser
treinados antes:
    python main.py --modelos knn neural_network random_forest ensemble
"""
import joblib
from sklearn.base import clone
from sklearn.ensemble import VotingClassifier

from src import config
from src.evaluation import executar

NOME = "ensemble"
MEMBROS = {"rf": "random_forest", "mlp": "neural_network", "knn": "knn"}
OPCOES_PREP = {"sem_prep": True}


class VotacaoComLimiar(VotingClassifier):
    """Votação "soft" que responde diabetes quando a probabilidade média
    ponderada passa de `limiar`."""

    def __init__(self, estimators, *, voting="soft", weights=None, n_jobs=None,
                 flatten_transform=True, verbose=False, limiar=0.5):
        super().__init__(estimators, voting=voting, weights=weights, n_jobs=n_jobs,
                         flatten_transform=flatten_transform, verbose=verbose)
        self.limiar = limiar

    def predict(self, X):
        prob_diabetes = self.predict_proba(X)[:, 1]
        return self.classes_[(prob_diabetes > self.limiar).astype(int)]


def _carregar_membro(nome):
    caminho = config.DIR_ARTEFATOS / f"{nome}.joblib"
    if not caminho.exists():
        raise FileNotFoundError(
            f"{caminho} não existe. Treine antes: python main.py --modelos {nome}")
    return clone(joblib.load(caminho))


def criar_modelo():
    membros = [(chave, _carregar_membro(nome)) for chave, nome in MEMBROS.items()]
    return VotacaoComLimiar(membros, voting="soft")


GRADE = {
    "modelo__weights": [[1, 1, 1], [2, 2, 1], [2, 1, 1], [1, 2, 1], [1, 1, 0]],  # peso de rf, mlp, knn
    "modelo__limiar": [0.45, 0.5],
}


def main(aninhada: bool = False, sufixo: str = ""):
    return executar(NOME + sufixo, criar_modelo(), GRADE, aninhada=aninhada,
                    opcoes_prep=OPCOES_PREP)


if __name__ == "__main__":
    main()
