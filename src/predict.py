"""Previsões a partir de um modelo treinado.

O pipeline salvo já contém a limpeza, então recebe o CSV bruto.
"""
from pathlib import Path

import joblib
import pandas as pd

from src import config
from src.preprocessing import ler_csv


def gerar_submissao(modelo, arquivo_entrada, arquivo_saida) -> pd.DataFrame:
    df = ler_csv(arquivo_entrada)
    previsao = modelo.predict(df)
    sub = pd.DataFrame({config.COL_ID: df[config.COL_ID], config.COL_ALVO: previsao.astype(int)})

    assert len(sub) == len(df), "número de linhas diferente da entrada"
    assert set(sub[config.COL_ALVO].unique()) <= {0, 1}

    Path(arquivo_saida).parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(arquivo_saida, index=False)
    print(f"  Submissão: {Path(arquivo_saida).name} ({len(sub)} linhas | "
          f"diabetes previstos = {int(sub[config.COL_ALVO].sum())})")
    return sub


def prever_com_modelo_salvo(nome: str, arquivo_entrada, arquivo_saida=None) -> pd.DataFrame:
    caminho = config.DIR_ARTEFATOS / f"{nome}.joblib"
    if not caminho.exists():
        raise FileNotFoundError(f"{caminho} não existe. Treine antes: python main.py --modelos {nome}")
    modelo = joblib.load(caminho)
    if arquivo_saida is None:
        arquivo_saida = config.DIR_SUBMISSOES / f"previsao_{nome}_{Path(arquivo_entrada).stem}.csv"
    return gerar_submissao(modelo, arquivo_entrada, arquivo_saida)
