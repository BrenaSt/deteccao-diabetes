# Dicionário de Dados — Detecção de Diabetes

Projeto Mineração de Dados · UFU · Brena Stephany, Jessica Lara e Ana Luiza Madeira · 09/11/2026

## Visão geral

A base tem **21 colunas**: 20 descrevem o paciente e o exame, e a coluna `class` diz se ele tem diabetes (1) ou é controle (0). Das 20, **11 entram nos modelos**, e `EXAM_DATE` e `BIRTH_YEAR` ajudam a recuperar a idade. O tratamento cria **7 colunas novas** (4 delas só para a Rede Neural). As demais servem para identificar ou foram retiradas.

| Arquivo | Linhas | Colunas | Observação |
| --- | --- | --- | --- |
| `train.csv` | 540 (520 sem duplicatas) | 21 | Tem a coluna `class` |
| `test.csv` | 312 | 20 | Sem `class`; nunca usado no treino |

**Como ler cada ficha:** *Tipo* é o tipo de dado depois do tratamento. *Como chegou* mostra os formatos e problemas encontrados no arquivo bruto. *Tratamento* é o que a função `limpar()` faz (em `src/preprocessing.py`). *No modelo* diz se e como a coluna entra nos modelos. *Relação com diabetes* compara controles e diabéticos.

Os números de vazios e valores brutos são do `train.csv` completo (540 linhas). Faixas, medianas e comparações por classe são do treino já limpo e sem duplicatas (520 pacientes). Não havia dicionário oficial: significados e unidades foram deduzidos dos valores.

## Resumo das colunas

As colunas mais vazias são SMOKER (33%) e HEMOGLOBIN (23%); a mais importante para os modelos é HEMOGLOBIN.

| Coluna | Significado | Grupo | Tipo (após tratamento) | Unidade | Vazio no bruto | No modelo |
| --- | --- | --- | --- | --- | --- | --- |
| `id` | Número da linha | Identificação | Inteiro | — | 0% | Não (só na saída) |
| `PATIENT_CODE` | Código do paciente | Identificação | Texto | — | 0% | Não (só para achar duplicatas) |
| `EXAM_DATE` | Data do exame | Coleta | Data | — | 0% | Indireto (ano, para a idade) |
| `COLLECTION_SITE` | Local da coleta | Coleta | Categórica | — | 0% | Não (retirada) |
| `LAB_DEVICE` | Aparelho do laboratório | Coleta | Categórica | — | 0% | Não (retirada) |
| `STUDY_PROTOCOL` | Protocolo do estudo | Coleta | Constante | — | 0% | Não (valor único) |
| `GENDER` | Sexo | Perfil | Categórica nominal | — | 1,1% | Sim |
| `AGE` | Idade | Perfil | Numérica | anos | 0,9% | Sim |
| `BIRTH_YEAR` | Ano de nascimento | Perfil | Numérica | ano | 1,9% | Indireto (recupera a idade) |
| `HEIGHT` | Altura | Perfil | Numérica | cm | 6,3% | Sim |
| `WEIGHT` | Peso | Perfil | Numérica | kg | 2,4% | Sim |
| `BLOOD_TYPE` | Tipo sanguíneo | Perfil | Categórica nominal | — | 3,9% | Não (retirada) |
| `SMOKER` | Tabagismo | Hábitos | Categórica nominal | — | 32,8% | Sim |
| `PHYSICAL_ACTIVITY` | Atividade física | Hábitos | Categórica ordinal | — | 4,8% | Sim |
| `FAMILY_HISTORY` | Diabetes na família | Hábitos | Categórica nominal | — | 3,5% | Sim |
| `GLUCOSE` | Glicose no sangue | Exames | Numérica | mg/dL | 11,3% | Sim |
| `HEMOGLOBIN` | Hemoglobina glicada (HbA1c) | Exames | Numérica | % | 23,0% | Sim |
| `TRIGLYCERIDES` | Triglicerídeos | Exames | Numérica | ln(1 + mg/dL) | 8,5% | Sim |
| `SODIUM` | Sódio no sangue | Exames | Numérica | mmol/L | 2,2% | Não (retirada) |
| `BLOOD_PRESSURE` | Pressão arterial | Exames | Texto → 2 numéricas | mmHg | 7,8% | Sim, como BP_SYS e BP_DIA |
| `class` | Diagnóstico | Alvo | Binária | 0 / 1 | 0% | É o que se prevê |

"Vazio no bruto" conta só campos em branco ou "?". Valores inválidos (como idade 999 ou "Unknown") também viram vazio na limpeza; por isso algumas colunas ficam mais vazias depois de tratadas.

## Identificação e coleta

Nenhuma destas colunas entra nos modelos: ou só identificam a linha, ou não têm relação com diabetes.

### id

- **Significado:** número da linha no arquivo. Único em cada linha.
- **Tipo:** inteiro. Sem vazios.
- **No modelo:** não. Volta só no arquivo de saída (`id`, `class`), para ligar cada previsão à sua linha.

### PATIENT_CODE

- **Significado:** código do paciente, no formato `PT-` + 6 caracteres (ex.: `PT-6AE951`).
- **Como chegou:** 540 linhas e 520 códigos diferentes. 20 códigos aparecem duas vezes.
- **Tratamento:** usado só para achar duplicatas no treino. Uma linha é duplicata quando código, todas as colunas limpas e `class` são iguais. As 20 repetições eram desse tipo e foram removidas (540 → 520).
- **No modelo:** não.

### EXAM_DATE

- **Significado:** data em que o exame foi feito. Anos entre 2018 e 2021.
- **Como chegou:** três formatos misturados.

| Formato | Exemplo | Linhas |
| --- | --- | --- |
| AAAA-MM-DD | 2019-07-05 | 313 |
| DD/MM/AAAA | 10/08/2020 | 165 |
| Mês por extenso | Jan 19 2021 | 62 |

- **Tratamento:** só o **ano** é extraído, o que evita a dúvida entre dia/mês e mês/dia.
- **No modelo:** indireto. O ano do exame menos `BIRTH_YEAR` recupera a idade quando `AGE` é inválida.

### COLLECTION_SITE

- **Significado:** local onde o sangue foi coletado.
- **Como chegou:** 52 locais (`SITE-001` a `SITE-060`), sem vazios. O maior, `SITE-001`, tem 141 pacientes; muitos têm 1 a 3.
- **Tratamento:** texto padronizado.
- **No modelo:** não (comentada em `config.py`). O local não causa diabetes, e locais com poucos pacientes só trazem ruído.

### LAB_DEVICE

- **Significado:** aparelho que analisou a amostra.
- **Como chegou:** `FTIR-A` (500 pacientes) e `FTIR-B` (20). Sem vazios.
- **Relação com diabetes:** nenhuma. 52% de diabéticos no FTIR-A e 55% no FTIR-B, praticamente o mesmo.
- **No modelo:** não (comentada em `config.py`).

### STUDY_PROTOCOL

- **Significado:** protocolo do estudo clínico.
- **Como chegou:** o mesmo valor em todas as linhas: `T2D-SPEC-v1` ("T2D" = diabetes tipo 2).
- **No modelo:** não. Uma coluna com um único valor não ajuda a separar ninguém.

## Perfil

Idade e peso são as colunas de perfil que mais separam os grupos: diabéticos têm mediana de 62 anos e 80 kg, contra 52 anos e 71 kg nos controles.

### GENDER

- **Significado:** sexo do paciente.
- **Tipo:** categórica nominal. Valores finais: `F` e `M`.
- **Como chegou:** 13 grafias diferentes.

| Valor bruto | Linhas | Vira |
| --- | --- | --- |
| FEMALE, "FEMALE ", Female, female, F, f | 305 + 2 + 7 + 5 + 11 + 9 | F |
| MALE, " MALE", Male, male, M, m | 173 + 8 + 2 + 5 + 3 + 4 | M |
| (em branco) | 6 | vazio |

- **Tratamento:** tira espaços, passa para maiúsculas e traduz pela lista de equivalências (`MAPA_GENERO`, que também aceita FEMININO, MULHER, MASCULINO, HOMEM). Após limpar: 326 F e 189 M.
- **Vazio:** vira `DESCONHECIDO`.
- **No modelo:** sim, em colunas sim/não (one-hot). Categorias com menos de 10 pacientes no treino se juntam em `GENDER_infrequent_sklearn`.
- **Relação com diabetes:** fraca. 50% de diabéticos entre mulheres e 55% entre homens.

### AGE

- **Significado:** idade no dia do exame.
- **Tipo:** numérica, em anos. Faixa aceita: 1 a 110.
- **Como chegou:** de −1 a 999. Valores impossíveis: 999 (3 linhas), 0 (2) e −1 (1). Em branco: 0,9%.
- **Tratamento:** idade inválida ou vazia vira **ano do exame − ano de nascimento**. Depois disso, nenhum paciente ficou sem idade.
- **Após limpar:** 16 a 98 anos, mediana 57.
- **Vazio:** mediana do treino (não ocorreu no treino, mas pode ocorrer em dados novos).
- **No modelo:** sim.
- **Relação com diabetes:** forte. Mediana de 62 anos nos diabéticos e 52 nos controles.

### BIRTH_YEAR

- **Significado:** ano de nascimento.
- **Tipo:** numérica. Faixa aceita: 1900 a 2015. No treino: 1920 a 2003. Em branco: 1,9%.
- **No modelo:** indireto. Só serve para recuperar a idade. Não entra sozinha porque repetiria a informação de `AGE`.

### HEIGHT

- **Significado:** altura.
- **Tipo:** numérica, em **cm**. Faixa aceita: 120 a 220.
- **Como chegou:** metros e centímetros misturados, de 1,47 a 187. **107 linhas** estavam em metros. Em branco: 6,3%.
- **Tratamento:** valor abaixo de 3 está em metros e é multiplicado por 100 (1,63 → 163).
- **Após limpar:** 143 a 187 cm, mediana 163.
- **No modelo:** sim (também entra no cálculo do IMC).
- **Relação com diabetes:** nenhuma sozinha (mediana 162 cm nos diabéticos e 163 cm nos controles).

### WEIGHT

- **Significado:** peso.
- **Tipo:** numérica, em **kg**. Faixa aceita: 30 a 250.
- **Como chegou:** 41,3 a 242,1 kg, todos na mesma unidade. Em branco: 2,4%.
- **Após limpar:** mediana 76,1 kg.
- **No modelo:** sim (também entra no IMC).
- **Relação com diabetes:** moderada. Mediana de 80,1 kg nos diabéticos e 71,4 kg nos controles.

### BLOOD_TYPE

- **Significado:** tipo sanguíneo.
- **Tipo:** categórica nominal: A+, A−, B+, B−, AB+, AB−, O+, O−.
- **Como chegou:** O+ 197, A+ 169, O− 51, B+ 48, A− 28, B− 14, AB+ 9, AB− 3. Em branco: 3,9%.
- **No modelo:** não (comentada em `config.py`). O tipo sanguíneo não tem relação conhecida com diabetes, e nos grupos grandes a proporção de diabéticos é parecida (47% a 54%). Tirá-la melhorou um pouco o KNN e a Rede Neural.

## Hábitos

Histórico familiar e atividade física são os hábitos ligados à doença: 69% de diabéticos entre quem tem diabetes na família (43% sem) e 62% entre sedentários (30% entre quem faz atividade intensa).

### SMOKER

- **Significado:** situação de tabagismo.
- **Tipo:** categórica nominal. Valores finais: `NEVER` (nunca fumou), `FORMER` (ex-fumante), `CURRENT` (fuma).
- **Como chegou:** 10 grafias e muitos vazios.

| Valor bruto | Linhas | Vira |
| --- | --- | --- |
| never, NEVER, Never | 181 + 16 + 7 | NEVER |
| former, Former, ex-smoker | 84 + 3 + 6 | FORMER |
| current, Current, yes | 58 + 4 + 4 | CURRENT |
| (em branco) | 177 (32,8%) | vazio |

- **Tratamento:** lista de equivalências `MAPA_FUMANTE` (também aceita NO, NUNCA, EX, SIM…). Após limpar: 196 NEVER, 89 FORMER, 65 CURRENT.
- **Vazio:** vira a categoria `DESCONHECIDO`, para não inventar que 1 em cada 3 pacientes nunca fumou.
- **No modelo:** sim, em colunas sim/não: SMOKER_NEVER, SMOKER_FORMER, SMOKER_CURRENT, SMOKER_DESCONHECIDO.
- **Relação com diabetes:** fraca e sem direção clara: 58% de diabéticos em NEVER, 46% em FORMER e 52% em CURRENT.

### PHYSICAL_ACTIVITY

- **Significado:** nível de atividade física.
- **Tipo:** categórica **ordinal**, codificada em ordem: SEDENTARY = 0, LIGHT = 1, MODERATE = 2, INTENSE = 3.
- **Como chegou:** 19 grafias, com erros de digitação.

| Valor bruto | Linhas | Vira |
| --- | --- | --- |
| Sedentary, "Sedentary ", SEDENTARY, sedentary, Sedentry | 149 + 10 + 6 + 6 + 5 | SEDENTARY (0) |
| Light, LIGHT, light, Ligth | 157 + 11 + 7 + 5 | LIGHT (1) |
| Moderate, moderate, " Moderate", MODERATE, Moderat | 99 + 4 + 3 + 3 + 1 | MODERATE (2) |
| Intense, INTENSE, intense, Intensive | 39 + 4 + 3 + 2 | INTENSE (3) |
| (em branco) | 26 (4,8%) | vazio |

- **Tratamento:** reconhece pelo **início da palavra** (SED, LIG ou LEV, MOD, INT), o que pega erros como "Ligth" e "Sedentry". Após limpar: 173 / 165 / 109 / 47.
- **Vazio:** recebe o valor mais frequente do treino, SEDENTARY. "DESCONHECIDO" não teria lugar na escala de 0 a 3, e os vazios não indicam a doença (50% de diabéticos, contra 52% no geral).
- **No modelo:** sim, como um número de 0 a 3 (escalado na Rede Neural e no KNN).
- **Relação com diabetes:** clara e em ordem: 62% de diabéticos em SEDENTARY, 56% em LIGHT, 39% em MODERATE e 30% em INTENSE.

### FAMILY_HISTORY

- **Significado:** se há diabetes na família.
- **Tipo:** categórica nominal. Valores finais: `YES` e `NO`.
- **Como chegou:** 14 grafias, em inglês, português e números.

| Valor bruto | Linhas | Vira |
| --- | --- | --- |
| No, NO, no, N, Não, 0 | 256 + 15 + 13 + 12 + 12 + 15 | NO |
| Yes, yes, YES, Y, Sim, 1 | 116 + 15 + 11 + 7 + 8 + 8 | YES |
| Unknown | 33 | vazio |
| (em branco) | 19 | vazio |

- **Tratamento:** lista de equivalências `MAPA_HISTORICO`. "Unknown" não é sim nem não, então vira vazio. Após limpar: 313 NO, 159 YES, 9,2% vazios.
- **Vazio:** vira a categoria `DESCONHECIDO`.
- **No modelo:** sim, em colunas sim/não: FAMILY_HISTORY_YES, FAMILY_HISTORY_NO, FAMILY_HISTORY_DESCONHECIDO.
- **Relação com diabetes:** forte. 69% de diabéticos entre quem tem histórico e 43% entre quem não tem.

## Exames

A HbA1c (coluna HEMOGLOBIN) é a coluna que mais separa os grupos: mediana de 7,6% nos diabéticos e 5,4% nos controles. Glicose e HbA1c chegaram com duas unidades misturadas.

### GLUCOSE

- **Significado:** glicose (açúcar) no sangue no momento do exame.
- **Tipo:** numérica, em **mg/dL**. Faixa aceita: 40 a 600.
- **Como chegou:** de 3,1 a 500. **45 linhas** estavam em mmol/L. Em branco: 11,3%.
- **Tratamento:** valor abaixo de 30 está em mmol/L e é convertido: mg/dL = mmol/L × 18,016 (5,6 → 101). Nenhuma glicose real em mg/dL fica abaixo de 30.
- **Após limpar:** 47 a 500 mg/dL, mediana 105.
- **Vazio:** mediana do treino.
- **No modelo:** sim. Também gera a coluna de corte GLIC_126.
- **Relação com diabetes:** forte. Mediana de 135 mg/dL nos diabéticos e 96 nos controles. Referência médica: 126 mg/dL ou mais em jejum indica diabetes.

### HEMOGLOBIN (HbA1c)

- **Significado:** apesar do nome, é a **hemoglobina glicada (HbA1c)**, a média do açúcar no sangue nos últimos 3 meses. Os valores (4% a 13%) são típicos de HbA1c; a hemoglobina comum fica entre 12 e 17 g/dL.
- **Tipo:** numérica, em **%**. Faixa aceita: 3 a 20.
- **Como chegou:** de 4,7 a 116. **44 linhas** estavam em mmol/mol (padrão IFCC). Em branco: 23,0%, a segunda coluna mais vazia.
- **Tratamento:** valor acima de 20 está em mmol/mol e é convertido pela equação oficial: % = mmol/mol ÷ 10,929 + 2,15 (48 → 6,5).
- **Após limpar:** 4,6% a 13,1%, mediana 6,2%.
- **Vazio:** mediana do treino (6,2%).
- **No modelo:** sim, e é a mais importante (62% das decisões da árvore). Também gera HB_65 e HB_57.
- **Relação com diabetes:** muito forte. Depois da conversão, **100% dos pacientes com HbA1c ≥ 6,5% são diabéticos**. Referência médica: abaixo de 5,7% normal; 5,7% a 6,4% pré-diabetes; 6,5% ou mais, diabetes.

### TRIGLYCERIDES

- **Significado:** triglicerídeos, a gordura no sangue.
- **Tipo:** numérica, guardada em **ln(1 + mg/dL)**. Faixa aceita antes do log: 10 a 5000 mg/dL.
- **Como chegou:** 21 a 2880 mg/dL, mediana 151. Em branco: 8,5%.
- **Tratamento:** aplica o logaritmo ln(1 + valor), que encolhe os extremos: 150 → 5,0 e 2880 → 8,0.
- **Após limpar:** 3,09 a 7,97 na escala log.
- **Vazio:** mediana do treino.
- **No modelo:** sim.
- **Relação com diabetes:** moderada. Mediana de cerca de 173 mg/dL nos diabéticos e 123 nos controles.

### SODIUM

- **Significado:** sódio no sangue.
- **Tipo:** numérica, em mmol/L. Faixa aceita: 120 a 160.
- **Como chegou:** 133 a 149, mediana 140. Em branco: 2,2%.
- **No modelo:** não (comentada em `config.py`). Mediana de 140 nos dois grupos: não separa diabéticos de controles.

### BLOOD_PRESSURE

- **Significado:** pressão arterial, máxima (sistólica) sobre mínima (diastólica).
- **Tipo:** texto no bruto; vira duas colunas numéricas em **mmHg** (BP_SYS e BP_DIA, na próxima seção).
- **Como chegou:**

| Formato | Exemplo | Linhas |
| --- | --- | --- |
| a/b | 120/80 | 446 |
| a x b | 123x89 | 27 |
| a / b (com espaços) | 114 / 75 | 25 |
| "?" | ? | 19 |
| (em branco) |  | 23 |

- **Tratamento:** separa os dois números por `/`, `x` ou `-`. "?" e branco viram vazio. Se a máxima for menor ou igual à mínima, os dois viram vazio (valores trocados ou errados).
- **No modelo:** sim, como BP_SYS e BP_DIA.

## Colunas criadas no tratamento

A limpeza cria 7 colunas. BMI, BP_SYS e BP_DIA entram em todos os modelos; as 4 colunas de corte clínico entram só na Rede Neural.

| Coluna | Como é calculada | Tipo / unidade | Faixa no treino | Mediana controle → diabetes | Usada por |
| --- | --- | --- | --- | --- | --- |
| `BMI` (IMC) | WEIGHT ÷ (HEIGHT ÷ 100)²; aceita 10 a 80 | Numérica, kg/m² | 16,5 a 78,2 (mediana 28,2) | 27,0 → 29,6 | Todos |
| `BP_SYS` | 1º número de BLOOD_PRESSURE; aceita 70 a 250 | Numérica, mmHg | 90 a 175 (mediana 124) | 117 → 128 | Todos |
| `BP_DIA` | 2º número de BLOOD_PRESSURE; aceita 40 a 150 | Numérica, mmHg | 53 a 107 (mediana 81) | 80 → 83 | Todos |
| `HB_65` | 1 se HbA1c ≥ 6,5% (critério de diabetes) | Binária 0/1 | 0 ou 1 | 0 → 1 | Só Rede Neural |
| `HB_57` | 1 se HbA1c ≥ 5,7% (pré-diabetes) | Binária 0/1 | 0 ou 1 | 0 → 1 | Só Rede Neural |
| `GLIC_126` | 1 se glicose ≥ 126 mg/dL (critério de diabetes) | Binária 0/1 | 0 ou 1 | 0 → 1 | Só Rede Neural |
| `N_EXAMES_AUSENTES` | Quantos entre HbA1c e glicose vieram vazios | Contagem | 0, 1 ou 2 | 0 → 0 | Só Rede Neural |

**Vazios:** BMI fica vazio quando falta altura ou peso (8,8%); BP_SYS e BP_DIA quando a pressão falta ou é inválida (7,5%). HB_65, HB_57 e GLIC_126 ficam vazias quando o exame de origem falta, e N_EXAMES_AUSENTES registra essa falta. Todos os vazios numéricos recebem a mediana do treino.

**Por que os cortes só na Rede Neural:** os critérios são os da Associação Americana de Diabetes (ADA). A Rede Neural, com só 2 neurônios, tem dificuldade de aprender sozinha uma regra em degrau como "6,5% ou mais"; com as colunas, a nota F1 subiu de 0,896 para 0,905. A Random Forest já aprende esses cortes sozinha, e no KNN eles pioraram a distância entre pacientes parecidos.

## Alvo: class

A base é quase equilibrada: depois de tirar as duplicatas, 52% dos pacientes são diabéticos e 48% são controles.

- **Significado:** diagnóstico do paciente. `1` = diabetes; `0` = controle (sem diabetes).
- **Tipo:** binária (0 ou 1). Sem vazios.
- **Onde existe:** só no `train.csv`. O `test.csv` vem sem ela, e o modelo a preenche.

| Base | Controle (0) | Diabetes (1) | Total |
| --- | --- | --- | --- |
| `train.csv` bruto | 259 | 281 | 540 |
| Treino sem duplicatas | 250 | 270 | 520 |

- **Na saída:** os arquivos de previsão (`submissions/` e `--prever`) têm duas colunas, `id` e `class`.
- **Por que o equilíbrio importa:** com os grupos quase do mesmo tamanho, o modelo não aprende a "chutar" sempre o grupo maior, e o F1-macro dá o mesmo peso aos dois.
