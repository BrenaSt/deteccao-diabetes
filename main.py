"""Ponto de entrada do projeto.

  python main.py                                   # treina KNN, árvore e rede neural
  python main.py --modelos random_forest ensemble  # modelos específicos
  python main.py --aninhada                        # + validação cruzada aninhada
  python main.py --sufixo _v2                      # salva os arquivos com outro nome
  python main.py --prever ensemble --entrada data/novo.csv --saida previsoes.csv
"""
import argparse
import warnings

from src.models import decision_tree, ensemble, knn, neural_network, random_forest
from src.predict import prever_com_modelo_salvo

MODELOS = {
    "knn": knn,
    "decision_tree": decision_tree,
    "neural_network": neural_network,
    "random_forest": random_forest,
    "ensemble": ensemble,
}
PADRAO = ["knn", "decision_tree", "neural_network"]


def comparar(resultados: dict) -> None:
    print(f"\n{'=' * 64}\n  COMPARAÇÃO (validação cruzada, média dos folds)\n{'=' * 64}")
    metricas = ["f1_macro", "precisao_macro", "recall_diabetes", "recall_controle"]
    print(f"  {'modelo':<16}" + "".join(f"{m:>17}" for m in metricas))
    for nome, r in resultados.items():
        vc = r["validacao_cruzada"]
        print(f"  {nome:<16}" + "".join(f"{vc[m]['media']:>17.4f}" for m in metricas))


def main():
    p = argparse.ArgumentParser(description="Detecção de diabetes")
    p.add_argument("--modelos", nargs="+", choices=list(MODELOS), default=PADRAO)
    p.add_argument("--aninhada", action="store_true",
                   help="também roda a validação cruzada aninhada (mais lenta)")
    p.add_argument("--sufixo", default="",
                   help="acrescenta ao nome dos arquivos gerados, para não sobrescrever")
    p.add_argument("--prever", metavar="NOME_DO_MODELO",
                   help="usa um modelo já treinado para prever um CSV")
    p.add_argument("--entrada", help="CSV bruto para --prever")
    p.add_argument("--saida", help="CSV de saída para --prever (opcional)")
    args = p.parse_args()

    if args.prever:
        if not args.entrada:
            p.error("--prever exige --entrada")
        prever_com_modelo_salvo(args.prever, args.entrada, args.saida)
        return

    warnings.filterwarnings("ignore", category=UserWarning)
    resultados = {n + args.sufixo: MODELOS[n].main(aninhada=args.aninhada, sufixo=args.sufixo) for n in args.modelos}
    if len(resultados) > 1:
        comparar(resultados)


if __name__ == "__main__":
    main()
