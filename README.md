# Detecção de Diabetes — KNN, Árvore de Decisão e Rede Neural

Projeto da disciplina **Mineração de Dados** do **8º período do curso de Gestão da Informação da Universidade Federal de Uberlândia (UFU)**.

**Alunas:** Brena Stephany, Jessica Lara e Ana Luiza Madeira.

Classificação de pacientes em **controle (0)** ou **diabetes (1)** a partir de dados clínicos. Os modelos são avaliados por Precisão, Recall (Diabetes), Recall (Controle) e F1-macro.

## Modelos

| Modelo | Arquivo | F1-macro (validação cruzada) | Modelo treinado no repositório? |
| --- | --- | --- | --- |
| **Rede Neural (MLP) — modelo final** | `src/models/neural_network.py` | 0,919 | Sim |
| Random Forest | `src/models/random_forest.py` | 0,914 | Sim |
| Árvore de Decisão | `src/models/decision_tree.py` | 0,885 | Sim |
| KNN | `src/models/knn.py` | 0,902 | **Não** (precisa treinar com os dados) |
| Ensemble (Random Forest + Rede Neural + KNN) | `src/models/ensemble.py` | 0,923 | **Não** (precisa treinar com os dados) |

O tratamento dos dados e a configuração de cada modelo estão descritos em [`docs/TRATAMENTO_DE_DADOS.md`](docs/TRATAMENTO_DE_DADOS.md).

### Por que o KNN e o Ensemble não vêm treinados

A base de dados é de uso exclusivo da disciplina e não pode ser publicada. O KNN não aprende regras nem pesos: ele **guarda os pacientes de treino** e, para cada paciente novo, procura os mais parecidos entre eles. Por isso o arquivo `artifacts/knn.joblib` contém a base de treino inteira. O Ensemble usa o KNN dentro dele, então `artifacts/ensemble.joblib` também contém a base.

Os outros três modelos guardam só o que aprenderam (pesos da rede, perguntas das árvores, medianas e escalas), sem nenhum paciente, e por isso estão no repositório prontos para uso.

## Estrutura

```
.
├── main.py                  # ponto de entrada: treina, avalia e gera previsões
├── requirements.txt         # dependências (versão exata do scikit-learn)
├── data/                    # train.csv e test.csv (NÃO incluídos no repositório)
├── docs/                    # documentação do tratamento de dados
├── src/
│   ├── config.py            # caminhos, semente e colunas usadas
│   ├── preprocessing.py     # limpeza e pré-processamento
│   ├── evaluation.py        # validação cruzada, métricas e arquivos de saída
│   ├── predict.py           # previsões com um modelo treinado
│   └── models/              # um arquivo por modelo
├── reports/                 # métricas de cada modelo
├── submissions/             # previsões do test.csv
└── artifacts/               # modelos treinados (só Rede Neural, Random Forest e Árvore no repositório)
```

O que cada pasta gerada contém:

- `artifacts/<modelo>.joblib`: o modelo treinado, com a limpeza e o pré-processamento incluídos. É o arquivo que o `--prever` carrega.
- `reports/<modelo>_metricas.json`: notas da validação cruzada e matriz de confusão. `reports/<modelo>_busca.csv`: todas as combinações de configurações testadas, com a nota de cada uma.
- `submissions/submission_<modelo>.csv`: previsões do `test.csv` no formato da competição.

## Passo a passo

Os comandos abaixo funcionam no Linux (incluindo Fedora) e no macOS. No Windows, veja as diferenças indicadas.

### 1. Instalar o Python e o Git

É preciso ter o **Python 3.11 ou mais novo** (exigência do scikit-learn 1.8.0) e o **Git**. No Fedora:

```bash
sudo dnf install -y python3 python3-pip git
```

No Windows, instale pelos sites oficiais: python.org e git-scm.com.

### 2. Baixar o projeto (git clone)

```bash
git clone <LINK_DO_REPOSITORIO>
cd <NOME_DA_PASTA_DO_REPOSITORIO>
```

### 3. Criar o ambiente virtual e instalar as dependências

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

No Windows (PowerShell), troque as duas primeiras linhas por:

```powershell
python -m venv .venv
.venv\Scripts\activate
```

Instale exatamente as versões do `requirements.txt`. Os modelos salvos foram gerados com o scikit-learn 1.8.0, e outra versão pode mostrar avisos ou dar resultados diferentes.

### 4. Escolher o caminho

| Quero… | Precisa dos dados em `data/`? | Vá para |
| --- | --- | --- |
| Usar a Rede Neural, a Random Forest ou a Árvore já treinadas | Não | Passo 5A |
| Usar o KNN ou o Ensemble | **Sim** | Passo 5B |
| Treinar e avaliar tudo do zero | **Sim** | Passo 5C |

O arquivo para prever (`--entrada`) deve ter **as mesmas colunas do `test.csv`** e pode vir **sem nenhum tratamento**: a limpeza é feita pelo próprio modelo. O arquivo de saída tem as colunas `id` e `class` (0 = controle, 1 = diabetes).

### 5A. Rede Neural, Random Forest ou Árvore de Decisão (sem treinar)

Os modelos já estão em `artifacts/`. Basta rodar a previsão:

```bash
# Rede Neural (modelo final)
python main.py --prever neural_network --entrada caminho/do/novo_teste.csv --saida previsoes_rede_neural.csv

# Random Forest
python main.py --prever random_forest --entrada caminho/do/novo_teste.csv --saida previsoes_random_forest.csv

# Árvore de Decisão
python main.py --prever decision_tree --entrada caminho/do/novo_teste.csv --saida previsoes_arvore.csv
```

### 5B. KNN ou Ensemble (é preciso treinar antes)

**1. Coloque os dados da disciplina na pasta `data/`**, com exatamente estes nomes:

```
data/train.csv
data/test.csv
```

Os dois arquivos são obrigatórios: o `train.csv` é usado no treino e o `test.csv` gera o arquivo em `submissions/`. A pasta `data/` está no `.gitignore` e nunca vai para o repositório.

**2. Treine o KNN e depois o Ensemble**, nesta ordem (o Ensemble usa o KNN, a Rede Neural e a Random Forest salvos em `artifacts/`):

```bash
python main.py --modelos knn ensemble
```

Leva menos de um minuto. Os arquivos `artifacts/knn.joblib` e `artifacts/ensemble.joblib` são criados na sua máquina e continuam fora do repositório.

**3. Rode a previsão:**

```bash
# KNN
python main.py --prever knn --entrada caminho/do/novo_teste.csv --saida previsoes_knn.csv

# Ensemble
python main.py --prever ensemble --entrada caminho/do/novo_teste.csv --saida previsoes_ensemble.csv
```

### 5C. Treinar e avaliar todos os modelos do zero

Com os dados em `data/` (veja o passo 5B, item 1):

```bash
python main.py --modelos knn decision_tree neural_network random_forest ensemble
```

O Ensemble vem por último porque usa os outros modelos já treinados. Cada modelo mostra as notas no terminal e gera os arquivos em `artifacts/`, `reports/` e `submissions/`. Treinar de novo sobrescreve os modelos salvos de Rede Neural, Random Forest e Árvore.

## Outras opções

Todas estas opções treinam modelos, então precisam dos dados em `data/` (passo 5B, item 1).

```bash
python main.py                          # treina só KNN, árvore e rede neural
python main.py --modelos neural_network # treina um modelo específico
python main.py --aninhada               # inclui a validação cruzada aninhada (mais lenta)
python main.py --sufixo _v2             # salva os arquivos com outro nome, sem sobrescrever
```
