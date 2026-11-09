"""Tratamento dos dados.

Fase A - limpar(df): regras FIXAS (não aprendem nada dos dados). Pode ser
         aplicada igualmente em treino, teste e conjunto externo.
Fase B - construir_preprocessador(): etapas que APRENDEM com o treino
         (mediana, escala, categorias). Ficam dentro do Pipeline do
         scikit-learn, então a validação cruzada as recalcula a cada fold.
"""
import re

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (FunctionTransformer, OneHotEncoder,
                                   OrdinalEncoder, PowerTransformer, RobustScaler)

from src import config

def ler_csv(caminho) -> pd.DataFrame:
    """Lê tudo como texto: nenhuma conversão automática do pandas."""
    return pd.read_csv(caminho, dtype=str, keep_default_na=False, encoding="utf-8")


# Textos que significam "não informado"
VAZIOS_TEXTO = {"", "?", "-", "NA", "N/A", "NAN", "NONE", "NULL", "UNKNOWN",
                "DESCONHECIDO", "<NA>"}

def _texto(s: pd.Series) -> pd.Series:
    """Remove espaços, coloca em MAIÚSCULAS e transforma 'não informado' em NaN."""
    s = s.astype(str).str.strip().str.upper()
    return s.where(~s.isin(VAZIOS_TEXTO), np.nan)

def _numero(s: pd.Series) -> pd.Series:
    """Converte texto em número (aceita vírgula decimal). Inválido -> NaN."""
    s = s.astype(str).str.strip().str.replace(",", ".", regex=False)
    return pd.to_numeric(s, errors="coerce")

def _faixa(s: pd.Series, minimo: float, maximo: float) -> pd.Series:
    """Valores fora da faixa fisicamente plausível -> NaN."""
    return s.where((s >= minimo) & (s <= maximo))

def _mapear(s: pd.Series, mapa: dict) -> pd.Series:
    """Padroniza categorias. Valor fora do dicionário -> NaN (protege
    contra variações novas no conjunto externo)."""
    return _texto(s).map(mapa)


MAPA_GENERO = {
    "F": "F", "FEMALE": "F", "FEM": "F", "FEMININO": "F", "MULHER": "F", "WOMAN": "F", "W": "F",
    "M": "M", "MALE": "M", "MASC": "M", "MASCULINO": "M", "HOMEM": "M", "MAN": "M",
}
MAPA_FUMANTE = {
    "NEVER": "NEVER", "NO": "NEVER", "N": "NEVER", "NUNCA": "NEVER", "NÃO": "NEVER", "NAO": "NEVER",
    "FORMER": "FORMER", "EX-SMOKER": "FORMER", "EX SMOKER": "FORMER", "EX": "FORMER", "EX-FUMANTE": "FORMER",
    "CURRENT": "CURRENT", "YES": "CURRENT", "Y": "CURRENT", "SIM": "CURRENT", "SMOKER": "CURRENT",
}
MAPA_HISTORICO = {
    "YES": "YES", "Y": "YES", "SIM": "YES", "S": "YES", "1": "YES", "TRUE": "YES",
    "NO": "NO", "N": "NO", "NÃO": "NO", "NAO": "NO", "0": "NO", "FALSE": "NO",
}
TIPOS_SANGUE = {t: t for t in ["A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-"]}

# Atividade física: casa pelo início da palavra, o que cobre erros de
# digitação (LIGTH, SEDENTRY, MODERAT, INTENSIVE...).
PREFIXOS_ATIVIDADE = [
    ("SED", "SEDENTARY"), ("LIG", "LIGHT"), ("LEV", "LIGHT"),
    ("MOD", "MODERATE"), ("INT", "INTENSE"),
]

def _atividade(s: pd.Series) -> pd.Series:
    s = _texto(s)
    def casar(v):
        if not isinstance(v, str):
            return np.nan
        for prefixo, valor in PREFIXOS_ATIVIDADE:
            if v.startswith(prefixo):
                return valor
        return np.nan
    return s.map(casar)

# Pressão: "120/80", "114 / 75", "123x89"
_RE_PRESSAO = re.compile(r"^\s*(\d+(?:[.,]\d+)?)\s*[/xX\\-]\s*(\d+(?:[.,]\d+)?)\s*$")

def _pressao(s: pd.Series) -> tuple[pd.Series, pd.Series]:
    partes = s.astype(str).str.extract(_RE_PRESSAO)
    sis = _faixa(_numero(partes[0]), 70, 250)
    dia = _faixa(_numero(partes[1]), 40, 150)
    invertida = sis.notna() & dia.notna() & (sis <= dia)   # sistólica deve ser maior
    return sis.mask(invertida), dia.mask(invertida)

def _ano_exame(s: pd.Series) -> pd.Series:
    """Extrai só o ano (evita a ambiguidade dd/mm x mm/dd)."""
    return _numero(s.astype(str).str.extract(r"((?:19|20)\d{2})")[0])


# Indicadores de corte clínico (0/1). Só entram no modelo que os listar
# no OPCOES_PREP (hoje, só a rede neural).
CORTES_CLINICOS = ["HB_65", "HB_57", "GLIC_126", "N_EXAMES_AUSENTES"]

TODAS_NUMERICAS = CORTES_CLINICOS + ["AGE", "HEIGHT", "WEIGHT", "BMI", "BP_SYS", "BP_DIA",
                   "GLUCOSE", "HEMOGLOBIN", "TRIGLYCERIDES", "SODIUM"]

def limpar(df: pd.DataFrame) -> pd.DataFrame:
    """Fase A: recebe o CSV bruto e devolve as colunas limpas.

    Não remove linhas nem usa a coluna 'class', então é aplicada do mesmo
    jeito no treino, no teste e em qualquer conjunto novo.
    """
    out = pd.DataFrame(index=df.index)

    # --- Idade: códigos inválidos (0, -1, 999) -> NaN; recupera pelo ano
    idade = _faixa(_numero(df["AGE"]), 1, 110)
    nascimento = _faixa(_numero(df["BIRTH_YEAR"]), 1900, 2015)
    idade_calc = _faixa(_ano_exame(df["EXAM_DATE"]) - nascimento, 1, 110)
    out["AGE"] = idade.fillna(idade_calc)

    # --- Altura: metros -> cm
    altura = _numero(df["HEIGHT"])
    altura = altura.where(altura >= 3, altura * 100)
    out["HEIGHT"] = _faixa(altura, 120, 220)

    out["WEIGHT"] = _faixa(_numero(df["WEIGHT"]), 30, 250)
    out["BMI"] = _faixa(out["WEIGHT"] / (out["HEIGHT"] / 100) ** 2, 10, 80)

    # --- Pressão arterial -> duas colunas
    out["BP_SYS"], out["BP_DIA"] = _pressao(df["BLOOD_PRESSURE"])

    # --- Glicose: mmol/L (< 30) -> mg/dL
    glic = _numero(df["GLUCOSE"])
    glic = glic.where(glic >= 30, glic * 18.016)
    out["GLUCOSE"] = _faixa(glic, 40, 600)

    # --- HbA1c: mmol/mol (> 20) -> %
    hb = _numero(df["HEMOGLOBIN"])
    hb = hb.where(hb <= 20, hb / 10.929 + 2.15)
    out["HEMOGLOBIN"] = _faixa(hb, 3, 20)

    # --- Triglicerídeos em log(1 + x), para reduzir o peso dos valores extremos
    out["TRIGLYCERIDES"] = np.log1p(_faixa(_numero(df["TRIGLYCERIDES"]), 10, 5000))

    out["SODIUM"] = _faixa(_numero(df["SODIUM"]), 120, 160)

    # --- Cortes clínicos da ADA como colunas 0/1: HbA1c >= 6,5% e glicose
    #     >= 126 mg/dL (diabetes), HbA1c >= 5,7% (pré-diabetes)
    hb_ok, gl_ok = out["HEMOGLOBIN"], out["GLUCOSE"]
    out["HB_65"] = (hb_ok >= 6.5).astype(float).where(hb_ok.notna())
    out["HB_57"] = (hb_ok >= 5.7).astype(float).where(hb_ok.notna())
    out["GLIC_126"] = (gl_ok >= 126).astype(float).where(gl_ok.notna())
    out["N_EXAMES_AUSENTES"] = (hb_ok.isna().astype(float) + gl_ok.isna().astype(float))

    # --- Categóricas
    out["PHYSICAL_ACTIVITY"] = _atividade(df["PHYSICAL_ACTIVITY"])
    out["GENDER"] = _mapear(df["GENDER"], MAPA_GENERO)
    out["SMOKER"] = _mapear(df["SMOKER"], MAPA_FUMANTE)
    out["FAMILY_HISTORY"] = _mapear(df["FAMILY_HISTORY"], MAPA_HISTORICO)
    out["BLOOD_TYPE"] = _mapear(df["BLOOD_TYPE"], TIPOS_SANGUE)
    out["LAB_DEVICE"] = _texto(df["LAB_DEVICE"])
    out["COLLECTION_SITE"] = _texto(df["COLLECTION_SITE"])

    # id, PATIENT_CODE, EXAM_DATE, BIRTH_YEAR, STUDY_PROTOCOL e class
    # não entram. Quais colunas o modelo usa é decidido no ColumnTransformer.
    num = [c for c in out.columns if c in TODAS_NUMERICAS]
    cat = [c for c in out.columns if c not in num]
    out[num] = out[num].astype(float)
    out[cat] = out[cat].astype(object).where(out[cat].notna(), np.nan)
    return out


def carregar_treino(verbose: bool = True):
    """Retorna (X bruto, y) do treino, sem duplicatas.

    X continua bruto porque a limpeza faz parte do pipeline salvo.
    """
    bruto = ler_csv(config.ARQ_TREINO)
    y = bruto[config.COL_ALVO].astype(int)

    # Duplicata = mesmo paciente, mesmos dados limpos e mesma classe.
    # Compara depois de limpar ("MALE" e " MALE" passam a ser iguais).
    chave = pd.concat([_texto(bruto[config.COL_PACIENTE]), limpar(bruto), y], axis=1)
    duplicada = chave.astype(str).duplicated(keep="first")

    pac = _texto(bruto[config.COL_PACIENTE])
    conflitos = y.groupby(pac).nunique()
    conflitos = conflitos[conflitos > 1]

    bruto, y = bruto[~duplicada].reset_index(drop=True), y[~duplicada].reset_index(drop=True)
    repetidos = _texto(bruto[config.COL_PACIENTE]).duplicated().sum()

    if verbose:
        print(f"Treino: {len(duplicada)} linhas | {duplicada.sum()} duplicatas removidas | "
              f"{len(bruto)} restantes")
        print(f"Classes: controle={int((y == 0).sum())}  diabetes={int((y == 1).sum())}")
        if len(conflitos):
            print(f"ATENÇÃO: {len(conflitos)} paciente(s) com classes diferentes: {list(conflitos.index)}")
        if repetidos:
            print(f"ATENÇÃO: {repetidos} paciente(s) ainda repetidos com dados diferentes "
                  "(considere StratifiedGroupKFold)")
    return bruto, y


ESCALADORES = {
    "robust": RobustScaler,      # (x - mediana) / IQR
    "power": PowerTransformer,   # Yeo-Johnson: encurta caudas longas e padroniza
    "nenhum": None,              # sem normalização (modelos de árvore)
}


def construir_preprocessador(numericas=None, ordinais=None, nominais=None,
                             escalador: str = "robust") -> Pipeline:
    """Fase B: limpar() + ColumnTransformer, prontos para o CSV bruto.

    numericas / ordinais / nominais: colunas usadas (padrão: config.py).
    escalador: "robust", "power" ou "nenhum".
    Mediana, IQR e categorias são aprendidos só com o treino.
    """
    numericas = config.COLS_NUMERICAS if numericas is None else numericas
    ordinais = config.COLS_ORDINAIS if ordinais is None else ordinais
    nominais = config.COLS_NOMINAIS if nominais is None else nominais

    def _escala():
        classe = ESCALADORES[escalador]
        return "passthrough" if classe is None else classe()

    numericas_pipe = Pipeline([
        ("imputar", SimpleImputer(strategy="median")),
        ("escalar", _escala()),
    ])
    ordinal_pipe = Pipeline([
        ("imputar", SimpleImputer(strategy="most_frequent")),
        ("codificar", OrdinalEncoder(categories=[ordinais[c] for c in ordinais])),
        ("escalar", _escala()),
    ])
    nominais_pipe = Pipeline([
        ("imputar", SimpleImputer(strategy="constant", fill_value="DESCONHECIDO")),
        ("onehot", OneHotEncoder(handle_unknown="infrequent_if_exist",
                                 min_frequency=config.MIN_FREQ_CATEGORIA,
                                 sparse_output=False)),
    ])

    transformadores = [("num", numericas_pipe, list(numericas))]
    if ordinais:
        transformadores.append(("ord", ordinal_pipe, list(ordinais)))
    if nominais:
        transformadores.append(("nom", nominais_pipe, list(nominais)))
    colunas = ColumnTransformer(transformadores, remainder="drop",
                                verbose_feature_names_out=False)

    return Pipeline([
        ("limpeza", FunctionTransformer(limpar)),
        ("colunas", colunas),
    ])
