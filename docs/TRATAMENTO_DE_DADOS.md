# Tratamento de Dados e Configuração dos Modelos — Detecção de Diabetes

Projeto da disciplina Mineração de Dados (8º período, Gestão da Informação, UFU) · Brena Stephany, Jessica Lara e Ana Luiza Madeira

## Visão geral

O tratamento transforma o CSV bruto (21 colunas, texto sujo, unidades misturadas) em atributos numéricos prontos para os modelos, e roda **dentro do pipeline salvo**: o modelo recebe o `test.csv` ou qualquer conjunto novo sem tratamento manual.

| Fase | O que faz | Aprende com os dados? | Onde roda | Função |
| --- | --- | --- | --- | --- |
| **A — Limpeza** | Padroniza texto, converte unidades, marca valores inválidos como vazio, cria colunas derivadas | Não (regras fixas) | Treino, teste e conjuntos novos, do mesmo jeito | `limpar()` |
| **Duplicatas** | Remove linhas repetidas | Não | **Só no treino** | `carregar_treino()` |
| **B — Preparação** | Preenche vazios, normaliza, transforma categorias em números | Sim — **só com o treino** | Dentro do `Pipeline`, recalculada a cada fold da validação cruzada | `construir_preprocessador()` |

Arquivos: `src/config.py` (colunas usadas), `src/preprocessing.py` (as duas fases) e `src/models/*.py` (opções próprias de cada modelo, no dicionário `OPCOES_PREP`).

## Fluxo completo

1. **Leitura** (`ler_csv`): tudo como texto, sem conversões automáticas do pandas.
2. **Limpeza — Fase A** (`limpar`):
    1. Remove espaços e coloca o texto em MAIÚSCULAS.
    2. Textos de "não informado" viram vazio (`?`, `NA`, `UNKNOWN`...).
    3. Converte números (vírgula decimal vira ponto; o que não é número vira vazio).
    4. Converte unidades (altura, glicose, HbA1c).
    5. Valores fora da faixa plausível viram vazio.
    6. Padroniza categorias com dicionários.
    7. Cria colunas derivadas: `BP_SYS`, `BP_DIA`, `BMI`, idade recalculada e cortes clínicos.
3. **Remoção de duplicatas** (só no treino): 540 → 520 linhas.
4. **Seleção de colunas** (`ColumnTransformer`): entram as colunas do `config.py` (e, na rede neural, os cortes clínicos).
5. **Preparação — Fase B**: mediana nos vazios numéricos, moda na atividade física, `DESCONHECIDO` nas categorias; normalização própria de cada modelo; one-hot.
6. **Modelo**.

As etapas 2, 4, 5 e 6 ficam juntas em `artifacts/<modelo>.joblib`.

## Fase A — Limpeza coluna a coluna

### Colunas numéricas

| Coluna | Conversão de unidade | Faixa válida | Outros tratamentos |
| --- | --- | --- | --- |
| `AGE` | — | 1 a 110 anos | `0`, `-1` e `999` viram vazio; se vazia, recalcula pelo ano do exame − `BIRTH_YEAR` |
| `HEIGHT` | Valor < 3 está em metros: × 100 | 120 a 220 cm | — |
| `WEIGHT` | — | 30 a 250 kg | — |
| `BMI` (nova) | — | 10 a 80 | Peso / (altura em metros)² |
| `BP_SYS` / `BP_DIA` (novas) | — | 70–250 / 40–150 | Extraídas de `BLOOD_PRESSURE` (`120/80`, `114 / 75`, `123x89`). Sistólica ≤ diastólica → as duas viram vazio |
| `GLUCOSE` | Valor < 30 está em mmol/L: × 18,016 | 40 a 600 mg/dL | — |
| `HEMOGLOBIN` (HbA1c) | Valor > 20 está em mmol/mol: valor / 10,929 + 2,15 | 3 a 20 % | Após a conversão, todos os pacientes do treino com HbA1c ≥ 6,5% são diabéticos |
| `TRIGLYCERIDES` | — | 10 a 5000 mg/dL | Transformado em log(1 + x) |
| `SODIUM` | — | 120 a 160 | Limpa, mas não usada por nenhum modelo |

### Cortes clínicos (novas colunas 0/1)

| Coluna | Regra | Critério |
| --- | --- | --- |
| `HB_65` | HbA1c ≥ 6,5% | Diagnóstico de diabetes (ADA) |
| `HB_57` | HbA1c ≥ 5,7% | Pré-diabetes (ADA) |
| `GLIC_126` | Glicose ≥ 126 mg/dL | Diagnóstico de diabetes (ADA) |
| `N_EXAMES_AUSENTES` | Quantos de HbA1c e glicose estão vazios (0, 1 ou 2) | — |

Só a rede neural usa essas colunas: elas melhoraram a rede e não ajudaram os outros modelos.

### Colunas categóricas

| Coluna | Valores finais | Mapeamento |
| --- | --- | --- |
| `GENDER` | `F`, `M` | `F`, `FEMALE`, `FEMININO`, `MULHER`... → `F`; `M`, `MALE`, `MASCULINO`, `HOMEM`... → `M` |
| `SMOKER` | `NEVER`, `FORMER`, `CURRENT` | `NO`, `NUNCA` → `NEVER`; `EX-SMOKER`, `EX` → `FORMER`; `YES`, `SIM` → `CURRENT` |
| `FAMILY_HISTORY` | `YES`, `NO` | `Y`, `SIM`, `1` → `YES`; `N`, `NÃO`, `0` → `NO`; `Unknown` → vazio |
| `PHYSICAL_ACTIVITY` | `SEDENTARY`, `LIGHT`, `MODERATE`, `INTENSE` | Pelo início da palavra (`SED…`, `LIG…`, `MOD…`, `INT…`), o que corrige erros como `Ligth` e `Sedentry` |

Valores fora dos dicionários viram vazio, então variações novas num conjunto externo não quebram o código. `BLOOD_TYPE`, `LAB_DEVICE` e `COLLECTION_SITE` também são limpas, mas não são usadas.

### Colunas descartadas

`id` (guardado para a submissão), `PATIENT_CODE` (usado só para achar duplicatas), `EXAM_DATE` e `BIRTH_YEAR` (usadas só para recalcular a idade), `STUDY_PROTOCOL` (valor sempre igual), `BLOOD_PRESSURE` (substituída por `BP_SYS` e `BP_DIA`) e `class` (a resposta).

## Remoção de duplicatas

O treino tinha **20 linhas duplicadas** (o mesmo exame do mesmo paciente com outro `id`). Uma linha é duplicata quando `PATIENT_CODE`, todas as colunas limpas e `class` são iguais às de uma linha anterior. A comparação é feita depois da limpeza. Ficam **520 pacientes** (250 controle, 270 diabetes). O teste nunca perde linhas.

## Fase B — Vazios, normalização e categorias

| Grupo | Vazios | Transformação |
| --- | --- | --- |
| Numéricas | Mediana do treino | Normalização própria de cada modelo |
| Ordinal (`PHYSICAL_ACTIVITY`) | Valor mais frequente | SEDENTARY=0, LIGHT=1, MODERATE=2, INTENSE=3, depois normalização |
| Nominais | Categoria `DESCONHECIDO` | One-hot; categorias com menos de 10 pacientes são agrupadas, e categorias novas não causam erro |

| Normalização | Fórmula | Usada em |
| --- | --- | --- |
| `robust` | (valor − mediana) / IQR | Rede neural |
| `power` (Yeo-Johnson) | Curva parecida com log, que encurta caudas longas, e depois padroniza | KNN |
| `nenhum` | Valores originais | Árvore e Random Forest (modelos de árvore só usam a ordem dos valores) |

## Configuração final de cada modelo

| Modelo | Colunas | Normalização | Hiperparâmetros | F1-macro (`main.py`) | F1-macro (validação repetida) |
| --- | --- | --- | --- | --- | --- |
| KNN | config.py (20 atributos) | `power` | k = 61, `weights=distance`, distância Euclidiana | 0,902 | 0,896 |
| Árvore de Decisão | config.py (20 atributos) | `nenhum` | `max_depth=4`, `min_samples_leaf=10`, `ccp_alpha=0.005`, gini | 0,885 | 0,888 |
| Rede Neural (MLP) | config.py + cortes clínicos (24 atributos) | `robust` | Média de 5 redes, 1 camada com 2 neurônios, `tanh`, `alpha=3`, solver `lbfgs`, limiar 0,55 | 0,919 | 0,906 |
| Random Forest | config.py (20 atributos) | `nenhum` | 800 árvores, `min_samples_leaf=5`, `max_features=0.3` | 0,914 | 0,909 |
| Ensemble | Cada membro usa o próprio pré-processamento | — | Votação soft de RF, MLP e KNN; pesos e limiar escolhidos na busca (última execução: pesos 2/2/1, limiar 0,5) | 0,923 | ≈ 0,91 |

A coluna `main.py` vem da busca de hiperparâmetros em 5 folds e é um pouco otimista. A validação repetida (5 folds × 5 repetições = 25 divisões) é a estimativa mais confiável.

## Configurações (`src/config.py`)

| Configuração | Valor | O que controla |
| --- | --- | --- |
| `ARQ_TREINO` / `ARQ_TESTE` | `data/train.csv` / `data/test.csv` | Arquivos de entrada |
| `SEMENTE` | 42 | Mesmas divisões da validação cruzada para todos os modelos |
| `N_FOLDS` | 5 | Folds da validação cruzada |
| `COLS_NUMERICAS` / `COLS_ORDINAIS` / `COLS_NOMINAIS` | Ver acima | Colunas usadas por todos os modelos |
| `MIN_FREQ_CATEGORIA` | 10 | Mínimo de pacientes para uma categoria ter coluna própria no one-hot |

Cada modelo pode mudar o pré-processamento só para ele, no dicionário `OPCOES_PREP` do seu arquivo:

```python
OPCOES_PREP = {"escalador": "power"}                                        # knn.py
OPCOES_PREP = {"numericas": config.COLS_NUMERICAS + CORTES_CLINICOS}        # neural_network.py
```

Opções aceitas: `numericas`, `ordinais`, `nominais` e `escalador` (`"robust"`, `"power"` ou `"nenhum"`).

## Como remover colunas

**De todos os modelos:** comente a linha da coluna no `src/config.py` (coloque `#` no início) e rode `python main.py` de novo. Para voltar, remova o `#`. As colunas `SODIUM`, `BLOOD_TYPE`, `LAB_DEVICE` e `COLLECTION_SITE` foram removidas assim.

**De um modelo só:** passe a lista de colunas no `OPCOES_PREP` daquele modelo, por exemplo:

```python
OPCOES_PREP = {"escalador": "power",
               "numericas": ["AGE", "BMI", "GLUCOSE", "HEMOGLOBIN", "TRIGLYCERIDES"]}
```

| Para remover | Faça |
| --- | --- |
| `PHYSICAL_ACTIVITY` | `COLS_ORDINAIS = {}` |
| Pressão arterial | Comente `BP_SYS` **e** `BP_DIA` |
| Todas as nominais | `COLS_NOMINAIS = []` |

## Testes que não melhoraram os modelos

Todos avaliados com validação cruzada repetida; nenhum superou a configuração final por mais que o desvio entre divisões (cerca de ±0,02).

| Teste | Resultado |
| --- | --- |
| Manter `SODIUM`, `BLOOD_TYPE`, `LAB_DEVICE`, `COLLECTION_SITE` | Pior no KNN e na rede; igual nos modelos de árvore |
| Outros preenchimentos de vazios (média, KNNImputer, imputação iterativa, HbA1c estimada pela glicose com a fórmula ADAG) | Igual ou pior que a mediana em todos os modelos |
| Outras normalizações (standard, minmax, quantil) | Pior que `power` no KNN e que `robust` na rede |
| Cortes clínicos nos outros modelos | Sem ganho |
| Novos cortes (idade ≥ 45, IMC ≥ 25/30, triglicerídeos ≥ 150, pressão ≥ 130/85, glicose ≥ 100) | Sem ganho |
| Aumento de dados (ruído gaussiano, SMOTE) | Igual ou pior |

**Onde os modelos erram:** a maioria dos erros está em pacientes com HbA1c entre 5,7% e 6,4% (faixa de pré-diabetes) ou sem o exame de HbA1c. Nenhum tratamento recupera essa informação, o que explica o teto em torno de 0,90–0,91.

## Limitações e regras a validar

Não existe dicionário de dados oficial: o significado das colunas e as conversões de unidade foram deduzidos dos valores.

| Regra | Situação |
| --- | --- |
| `HEMOGLOBIN` é HbA1c e valores > 20 estão em mmol/mol | Forte evidência: após a conversão, HbA1c ≥ 6,5% = 100% diabéticos |
| Glicose < 30 está em mmol/L | Provável: os valores convertidos ficam na faixa normal de mg/dL |
| Peso > 180 pode estar em libras | Não convertido; converter piorou os modelos |
| `SMOKER = yes` significa fumante atual | Suposição (4 casos) |
