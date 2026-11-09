"""Random Forest.

Treina 800 árvores, cada uma com uma amostra sorteada dos pacientes e, a
cada pergunta, um sorteio de 30% das colunas. A resposta é a média dos
votos, bem mais estável que a de uma árvore sozinha.
"""
from sklearn.ensemble import RandomForestClassifier

from src import config
from src.evaluation import executar

NOME = "random_forest"
OPCOES_PREP = {"escalador": "nenhum"}


def criar_modelo():
    return RandomForestClassifier(n_estimators=800, random_state=config.SEMENTE, n_jobs=1)


GRADE = {
    "modelo__min_samples_leaf": [5],   # mínimo de pacientes por folha
    "modelo__max_features": [0.3],     # fração das colunas sorteadas em cada pergunta
    "modelo__max_depth": [None],
}


def main(aninhada: bool = False, sufixo: str = ""):
    return executar(NOME + sufixo, criar_modelo(), GRADE, aninhada=aninhada,
                    opcoes_prep=OPCOES_PREP)


if __name__ == "__main__":
    main()
