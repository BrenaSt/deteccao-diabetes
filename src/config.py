"""Configurações do projeto: caminhos, semente e colunas usadas pelos modelos.

Para tirar uma coluna de todos os modelos, comente a linha dela abaixo.
"""
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
DIR_DADOS = RAIZ / "data"
DIR_ARTEFATOS = RAIZ / "artifacts"      # modelos treinados (.joblib)
DIR_RELATORIOS = RAIZ / "reports"       # métricas e resultados da validação
DIR_SUBMISSOES = RAIZ / "submissions"   # previsões no formato do Kaggle

ARQ_TREINO = DIR_DADOS / "train.csv"
ARQ_TESTE = DIR_DADOS / "test.csv"

SEMENTE = 42
N_FOLDS = 5
COL_ID = "id"
COL_PACIENTE = "PATIENT_CODE"
COL_ALVO = "class"    # 0 = controle, 1 = diabetes

# Numéricas: mediana nos vazios -> normalização
COLS_NUMERICAS = [
    "AGE",
    "HEIGHT",
    "WEIGHT",
    "BMI",            # calculada: peso / altura²
    "BP_SYS",         # pressão sistólica, extraída de BLOOD_PRESSURE
    "BP_DIA",         # pressão diastólica, extraída de BLOOD_PRESSURE
    "GLUCOSE",
    "HEMOGLOBIN",
    "TRIGLYCERIDES",
    # "SODIUM",          # removida: pouca relação com diabetes
]

# Ordinais (categoria com ordem): moda nos vazios -> 0..3 -> normalização
COLS_ORDINAIS = {
    "PHYSICAL_ACTIVITY": ["SEDENTARY", "LIGHT", "MODERATE", "INTENSE"],
}

# Nominais (categoria sem ordem): "DESCONHECIDO" nos vazios -> one-hot
COLS_NOMINAIS = [
    "GENDER",
    "SMOKER",
    "FAMILY_HISTORY",
    # "BLOOD_TYPE",      # removida: pouca relação com diabetes
    # "LAB_DEVICE",      # removida: dado do processo de coleta, não do paciente
    # "COLLECTION_SITE", # removida: dado do processo de coleta, não do paciente
]

# No one-hot, categorias com menos pacientes que isso são agrupadas
MIN_FREQ_CATEGORIA = 10
