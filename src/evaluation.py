"""Treino, validação cruzada e geração de resultados, iguais para todos os modelos.

Cada arquivo em src/models/ define só o classificador, a grade de
hiperparâmetros e as opções de pré-processamento próprias.
"""
import json
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (classification_report, confusion_matrix,
                             make_scorer, precision_score, recall_score)
from sklearn.model_selection import (GridSearchCV, StratifiedKFold,
                                     cross_val_predict, cross_validate)
from sklearn.pipeline import Pipeline

from src import config
from src.preprocessing import carregar_treino, construir_preprocessador
from src.predict import gerar_submissao

METRICAS = {
    "f1_macro": "f1_macro",
    "precisao_macro": "precision_macro",
    "precisao_diabetes": make_scorer(precision_score, pos_label=1, zero_division=0),
    "recall_diabetes": make_scorer(recall_score, pos_label=1),
    "recall_controle": make_scorer(recall_score, pos_label=0),
    "acuracia": "accuracy",
}
METRICA_PRINCIPAL = "f1_macro"


def montar_pipeline(classificador, **opcoes_prep) -> Pipeline:
    """opcoes_prep vai para construir_preprocessador (colunas, escalador).
    Com sem_prep=True, o classificador já traz o próprio pré-processamento
    (caso do ensemble)."""
    opcoes_prep = dict(opcoes_prep)
    if opcoes_prep.pop("sem_prep", False):
        return Pipeline([("modelo", classificador)])
    return Pipeline([
        ("prep", construir_preprocessador(**opcoes_prep)),
        ("modelo", classificador),
    ])


def buscar(classificador, grade: dict, X, y, **opcoes_prep) -> GridSearchCV:
    """Busca de hiperparâmetros com validação cruzada estratificada."""
    busca = GridSearchCV(montar_pipeline(classificador, **opcoes_prep), grade,
                         scoring=METRICAS, refit=METRICA_PRINCIPAL,
                         cv=_folds(), n_jobs=-1, error_score="raise")
    return busca.fit(X, y)


def _folds(semente_extra: int = 0) -> StratifiedKFold:
    return StratifiedKFold(n_splits=config.N_FOLDS, shuffle=True,
                           random_state=config.SEMENTE + semente_extra)


def _resumo(cv_results: dict, idx: int) -> dict:
    return {m: {"media": float(cv_results[f"mean_test_{m}"][idx]),
                "desvio": float(cv_results[f"std_test_{m}"][idx])}
            for m in METRICAS}


def _imprimir(titulo: str, resumo: dict) -> None:
    print(f"\n  {titulo}")
    for m, v in resumo.items():
        print(f"    {m:<18} {v['media']:.4f} ± {v['desvio']:.4f}")


def executar(nome: str, classificador, grade: dict, aninhada: bool = False,
             opcoes_prep: dict = None) -> dict:
    """Busca de hiperparâmetros, validação cruzada, treino final e previsão do teste.

    grade       : hiperparâmetros a testar, com o prefixo "modelo__"
    aninhada    : também roda a validação cruzada aninhada (sem o otimismo da busca)
    opcoes_prep : pré-processamento próprio do modelo (None = config.py)
    """
    opcoes_prep = opcoes_prep or {}
    inicio = time.time()
    print(f"\n{'=' * 64}\n  MODELO: {nome.upper()}\n{'=' * 64}")
    X, y = carregar_treino()
    pipeline = montar_pipeline(classificador, **opcoes_prep)

    busca = buscar(classificador, grade, X, y, **opcoes_prep)
    melhor = busca.best_index_
    n_comb = len(busca.cv_results_["params"])
    print(f"\n  {n_comb} combinações testadas x {config.N_FOLDS} folds")
    print(f"  Melhores hiperparâmetros: {busca.best_params_}")
    resumo = _resumo(busca.cv_results_, melhor)
    _imprimir(f"Validação cruzada ({config.N_FOLDS} folds) - melhor combinação:", resumo)

    # Matriz de confusão com previsões fora do fold
    prev = cross_val_predict(busca.best_estimator_, X, y, cv=_folds(), n_jobs=-1)
    matriz = confusion_matrix(y, prev, labels=[0, 1])
    print("\n  Matriz de confusão (linhas = real, colunas = previsto)")
    print(f"                 controle  diabetes")
    print(f"    controle     {matriz[0, 0]:>8}  {matriz[0, 1]:>8}")
    print(f"    diabetes     {matriz[1, 0]:>8}  {matriz[1, 1]:>8}")
    print("\n" + classification_report(y, prev, target_names=["controle", "diabetes"],
                                       digits=4))

    # Validação aninhada: a busca roda dentro de cada fold externo
    resumo_aninhado = None
    if aninhada:
        busca_interna = GridSearchCV(pipeline, grade, scoring=METRICA_PRINCIPAL,
                                     cv=_folds(), n_jobs=-1)
        externo = cross_validate(busca_interna, X, y, scoring=METRICAS,
                                 cv=_folds(semente_extra=1), n_jobs=1)
        resumo_aninhado = {m: {"media": float(np.mean(externo[f"test_{m}"])),
                               "desvio": float(np.std(externo[f"test_{m}"]))}
                           for m in METRICAS}
        _imprimir("Validação ANINHADA (estimativa sem otimismo):", resumo_aninhado)

    # O refit já treinou o melhor modelo com todo o treino
    for d in (config.DIR_ARTEFATOS, config.DIR_RELATORIOS, config.DIR_SUBMISSOES):
        d.mkdir(exist_ok=True)
    joblib.dump(busca.best_estimator_, config.DIR_ARTEFATOS / f"{nome}.joblib")

    pd.DataFrame(busca.cv_results_).sort_values(f"rank_test_{METRICA_PRINCIPAL}") \
        .to_csv(config.DIR_RELATORIOS / f"{nome}_busca.csv", index=False)
    relatorio = {
        "modelo": nome,
        "melhores_hiperparametros": {k: str(v) for k, v in busca.best_params_.items()},
        "validacao_cruzada": resumo,
        "validacao_aninhada": resumo_aninhado,
        "matriz_confusao": matriz.tolist(),
        "n_treino": int(len(y)),
    }
    with open(config.DIR_RELATORIOS / f"{nome}_metricas.json", "w", encoding="utf-8") as f:
        json.dump(relatorio, f, indent=2, ensure_ascii=False)

    saida = config.DIR_SUBMISSOES / f"submission_{nome}.csv"
    gerar_submissao(busca.best_estimator_, config.ARQ_TESTE, saida)

    print(f"  Arquivos: artifacts/{nome}.joblib | reports/{nome}_*.json/csv | "
          f"submissions/{saida.name}")
    print(f"  Tempo: {time.time() - inicio:.1f}s")
    relatorio["modelo_treinado"] = busca.best_estimator_
    return relatorio
