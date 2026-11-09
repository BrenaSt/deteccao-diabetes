"""K-Vizinhos Mais Próximos (KNN).

Classifica um paciente pela classe dos K pacientes mais parecidos do treino.
Usa a normalização Power, que encurta as caudas longas de glicose e
triglicerídeos e deixa a distância entre pacientes mais justa.
"""
from sklearn.neighbors import KNeighborsClassifier

from src.evaluation import executar

NOME = "knn"
OPCOES_PREP = {"escalador": "power"}


def criar_modelo():
    return KNeighborsClassifier()


GRADE = {
    "modelo__n_neighbors": [21, 31, 41, 51, 61, 81],  # quantos vizinhos votam
    "modelo__weights": ["uniform", "distance"],       # "distance": o mais próximo pesa mais
    "modelo__p": [1, 2],                              # 1 = Manhattan, 2 = Euclidiana
    "prep__colunas__num__imputar__add_indicator": [False, True],  # coluna "estava vazio"
}


def main(aninhada: bool = False, sufixo: str = ""):
    return executar(NOME + sufixo, criar_modelo(), GRADE, aninhada=aninhada,
                    opcoes_prep=OPCOES_PREP)


if __name__ == "__main__":
    main()
