"""Árvore de Decisão.

Aprende uma sequência de perguntas do tipo "HbA1c <= 6,4?". Também salva
a árvore em texto e a importância das variáveis em
reports/decision_tree_regras.txt.
"""
from sklearn.tree import DecisionTreeClassifier, export_text

from src import config
from src.evaluation import executar

NOME = "decision_tree"

# A árvore só usa a ordem dos valores: sem normalização, o resultado é o
# mesmo e as regras saem em unidades reais.
OPCOES_PREP = {"escalador": "nenhum"}


def criar_modelo():
    return DecisionTreeClassifier(criterion="gini", random_state=config.SEMENTE)


GRADE = {
    "modelo__max_depth": [4],          # máximo de perguntas em sequência
    "modelo__min_samples_leaf": [10],  # mínimo de pacientes em cada folha
    "modelo__ccp_alpha": [0.005],      # poda: remove galhos que ajudam pouco
    "modelo__class_weight": [None],
}


def salvar_regras(pipeline, nome: str = NOME):
    arvore = pipeline.named_steps["modelo"]
    nomes = list(pipeline.named_steps["prep"].named_steps["colunas"].get_feature_names_out())
    texto = export_text(arvore, feature_names=nomes, decimals=2)
    importancias = sorted(zip(nomes, arvore.feature_importances_), key=lambda t: -t[1])
    with open(config.DIR_RELATORIOS / f"{nome}_regras.txt", "w", encoding="utf-8") as f:
        f.write(f"Profundidade: {arvore.get_depth()} | Folhas: {arvore.get_n_leaves()}\n\n")
        f.write("IMPORTÂNCIA DAS VARIÁVEIS\n")
        for n, v in importancias:
            if v > 0:
                f.write(f"  {n:<35} {v:.4f}\n")
        f.write("\nÁRVORE (unidades reais; TRIGLYCERIDES em log(1 + x))\n")
        f.write(texto)
    print(f"  Regras da árvore: reports/{nome}_regras.txt "
          f"(profundidade {arvore.get_depth()}, {arvore.get_n_leaves()} folhas)")


def main(aninhada: bool = False, sufixo: str = ""):
    resultado = executar(NOME + sufixo, criar_modelo(), GRADE, aninhada=aninhada,
                         opcoes_prep=OPCOES_PREP)
    salvar_regras(resultado["modelo_treinado"], NOME + sufixo)
    return resultado


if __name__ == "__main__":
    main()
