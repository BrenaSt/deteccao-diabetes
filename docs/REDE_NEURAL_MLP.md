# Rede Neural (MLP) — como ela decide

Projeto da disciplina **Mineração de Dados** (8º período de Gestão da Informação, UFU). Alunas: Brena Stephany, Jessica Lara e Ana Luiza Madeira.

Os números abaixo vêm do modelo salvo (`artifacts/neural_network.joblib`), da primeira das 5 redes, e de dois pacientes do conjunto de teste, chamados de **Paciente A** e **Paciente B**.

---

## 1. Visão geral

O modelo final são **5 redes pequenas iguais**. Cada uma dá uma chance de diabetes; a média das 5 é comparada com **0,55**.

```
24 entradas ──► 2 neurônios escondidos (tanh) ──► 1 neurônio de saída (sigmoide) ──► chance de diabetes
5 redes ──► média das 5 chances ──► média > 0,55 ? ──► diabetes (1) : controle (0)
```

Cada rede tem 53 números aprendidos: 48 pesos e 2 vieses na camada escondida, mais 2 pesos e 1 viés na saída.

## 2. O que entra na rede

A rede recebe 24 números já tratados: 9 numéricas, 4 colunas de corte (HB_65, HB_57, GLIC_126, N_EXAMES_AUSENTES), a atividade física (0 a 3) e 10 colunas sim/não de gênero, tabagismo e histórico familiar.

As numéricas passam pela escala **Robust**: (valor − mediana) ÷ IQR. Para a HbA1c, a mediana do treino é 6,2% e o IQR é 1,5. Então 5,0% entra como −0,80; 6,2% entra como 0; e 8,0% entra como +1,20.

---

## 3. O neurônio: pesos, viés e ativação

Todo neurônio faz duas contas:

```
z = w₁·x₁ + w₂·x₂ + … + wₙ·xₙ + b        (soma ponderada + viés)
saída = ativação(z)
```

### O viés (b)

O viés é aprendido no treino, como os pesos. Começa sorteado (entre ±0,48 na camada escondida e ±1,41 na saída, pela semente) e é ajustado a cada passo para diminuir o erro. O freio **alpha penaliza só os pesos**, então o viés fica livre.

Os pesos dizem **o quanto** cada entrada importa; o viés diz **a partir de que ponto** o neurônio "acende".

| Neurônio | Viés | Efeito |
| --- | --- | --- |
| Escondido 1 | −0,804 | Exigente: as entradas precisam somar mais de 0,804 para ele ficar positivo |
| Escondido 2 | +0,037 | Neutro |
| Saída | +0,353 | Sem sinal nenhum, a chance já seria 59%: leve inclinação para diabetes |

### A função de ativação

É uma fórmula fixa, escolhida antes do treino, que transforma o `z` (qualquer número) na resposta do neurônio.

| Função | Fórmula | Faixa | Onde | z = −2 | z = 0 | z = 2 |
| --- | --- | --- | --- | --- | --- | --- |
| tanh | (eᶻ − e⁻ᶻ) ÷ (eᶻ + e⁻ᶻ) | −1 a 1 | Escondidos | −0,96 | 0 | +0,96 |
| sigmoide | 1 ÷ (1 + e⁻ᶻ) | 0 a 1 | Saída | 0,12 | 0,50 | 0,88 |
| relu (testada) | maior entre 0 e z | 0 a ∞ | Empatou com a tanh | 0 | 0 | 2 |

**Por que ela é indispensável:** sem ativação, cada neurônio escondido devolveria uma soma ponderada das entradas, e a saída faria uma soma ponderada dessas somas. Somas de somas são só **outra soma**: a rede inteira viraria um único neurônio, e os escondidos não serviriam para nada. A curva da tanh "dobra" cada soma, e é essa dobra que permite criar o degrau e a rampa da seção 4.

A tanh também **satura**: z = −2,2 dá −0,975 e z = −5 dá −0,9999. Um paciente "muito normal" e um "extremamente normal" mandam quase o mesmo sinal, e nenhum valor extremo domina.

---

## 4. Os neurônios escondidos: um degrau e uma rampa

Cada neurônio escondido é um avaliador que dá uma nota de −1 (nada indica diabetes) a +1 (tudo indica), com seus próprios pesos. Essas notas são chamadas de **h₁** (neurônio 1) e **h₂** (neurônio 2), com "h" de *hidden*, "escondido" em inglês:

```
h₁ = tanh(soma das 24 entradas × pesos do neurônio 1 + viés −0,804)
h₂ = tanh(soma das 24 entradas × pesos do neurônio 2 + viés +0,037)
```

São só esses dois números que seguem para o neurônio de saída. A rede "resume" o paciente inteiro, de 24 colunas, em duas notas.

### Os pesos que mais pesam (rede 1)

| Entrada | Neurônio 1 | Neurônio 2 |
| --- | --- | --- |
| HB_65 (HbA1c ≥ 6,5%) | **+1,13** | +0,61 |
| HEMOGLOBIN (HbA1c contínua) | +0,73 | **+0,96** |
| HB_57 (HbA1c ≥ 5,7%) | −0,69 | +0,56 |
| GLUCOSE | +0,63 | +0,10 |
| TRIGLYCERIDES | +0,25 | +0,46 |
| AGE | +0,05 | +0,33 |

O neurônio 1 se apoia no **corte** de 6,5%. O neurônio 2 se apoia na HbA1c **contínua** e em fatores de risco (idade, triglicerídeos).

**Cuidado:** colunas que carregam a mesma informação (HbA1c, HB_57 e HB_65; peso, altura e IMC) dividem o efeito entre si, e um peso pode sair com sinal "estranho" para compensar outro, como o −0,69 da HB_57. Por isso o que vale é o efeito conjunto, mostrado abaixo.

### O que acontece quando só a HbA1c muda

Paciente A (glicose 124, 61 anos), mudando apenas a HbA1c:

| HbA1c | h₁ (neurônio 1) | h₂ (neurônio 2) | Chance (média das 5 redes) |
| --- | --- | --- | --- |
| 5,0% | −0,12 | −0,98 | 0,07 |
| 6,2% | −0,22 | −0,70 | 0,18 |
| 6,4% | −0,13 | −0,63 | 0,24 |
| **6,5%** | **+0,78** | −0,06 | **0,90** |
| 7,0% | +0,86 | +0,26 | 0,95 |
| 8,0% | +0,95 | +0,72 | 0,98 |

- **Neurônio 2 = rampa:** sobe aos poucos, de −0,98 a +0,72, conforme a HbA1c aumenta.
- **Neurônio 1 = degrau:** quase parado até 6,4% e salta para +0,78 em 6,5%. De 6,4% para 6,5%, a chance vai de 0,24 para 0,90.

Esse degrau vem da coluna HB_65. Com só 2 neurônios e curvas suaves, a rede dificilmente criaria sozinha um salto tão preciso, e é por isso que as colunas de corte existem.

### Por que só 2 neurônios

Com 520 pacientes, cada peso a mais é uma chance a mais de decorar o treino. Com o freio forte, redes de 4 e 8 neurônios deram exatamente as mesmas respostas: os neurônios extras só repetiam o trabalho. Um degrau e uma rampa bastam para o que os dados mostram.

---

## 5. O neurônio de saída e a sigmoide

O neurônio de saída recebe só h₁ e h₂, as notas dos dois neurônios escondidos (seção 4):

```
z = 2,524 × h₁ + 2,036 × h₂ + 0,353
chance = sigmoide(z) = 1 ÷ (1 + e⁻ᶻ)
```

Os dois pesos são positivos: quanto mais os escondidos "acendem", maior a chance. O degrau (h₁) tem um pouco mais de voz.

**Paciente A** (HbA1c 5,0%): h₁ = −0,121 e h₂ = −0,975.

```
z = 2,524 × (−0,121) + 2,036 × (−0,975) + 0,353 = −1,937
chance = 1 ÷ (1 + e^1,937) = 1 ÷ 7,94 = 0,126
```

**Paciente B** (HbA1c e glicose vazias): h₁ = +0,136 e h₂ = −0,116. Sem os dois exames, as pistas se anulam, e quem decide é o viés.

```
z = 2,524 × 0,136 + 2,036 × (−0,116) + 0,353 = 0,460
chance = 0,613
```

### O que a sigmoide faz

| z | −4 | −2 | 0 | **0,2** | 2 | 4 |
| --- | --- | --- | --- | --- | --- | --- |
| chance | 0,02 | 0,12 | 0,50 | **0,55** | 0,88 | 0,98 |

- Transforma qualquer `z` numa probabilidade: negativo = mais para controle; positivo = mais para diabetes.
- É sensível no meio e satura nos extremos, como a tanh. As duas são a mesma curva em "S": sigmoide(z) = (tanh(z/2) + 1) ÷ 2.
- O limiar de 0,55 equivale a exigir **z > 0,2**.
- Como h₁ e h₂ ficam entre −1 e 1, o `z` desta rede vai de −4,21 a +4,91: a chance nunca passa de 1,5% a 99,3%. A rede nunca diz 0% nem 100%.

**Por que um único neurônio de saída:** a chance de controle é 1 − chance de diabetes. O `predict_proba` devolve as duas, `[0,874; 0,126]` no Paciente A, e o código usa `[:, 1]` para pegar a de diabetes.

---

## 6. Cinco redes, média e limiar

Cada rede começa de pesos sorteados por uma semente diferente (42, 101, 202, 303, 404) e termina um pouco diferente. A média tira essa sorte:

| Paciente | Rede 1 | Rede 2 | Rede 3 | Rede 4 | Rede 5 | Média | Resposta |
| --- | --- | --- | --- | --- | --- | --- | --- |
| A | 0,126 | 0,060 | 0,059 | 0,057 | 0,059 | **0,072** | Controle |
| B | 0,613 | 0,531 | 0,533 | 0,749 | 0,532 | **0,592** | Diabetes |

No Paciente B, a rede 4 sozinha estava bem mais confiante (0,749); a média suaviza isso.

No código, `np.mean(..., axis=0)` faz essa média. As 5 tabelas de chances formam um bloco de formato (5 redes, pacientes, 2 colunas); `axis=0` tira a média ao longo das redes, paciente por paciente.

**O limiar de 0,55** decide qual erro a rede comete mais: mais alto, ela deixa passar diabéticos; mais baixo, acusa controles. Com 0,55 os acertos nos dois grupos ficaram equilibrados.

---

## 7. Como a rede aprende

1. Os pesos e vieses começam sorteados.
2. A rede calcula a chance de cada paciente de treino.
3. Mede o erro com a **log-loss**: −ln(chance dada à resposta certa). Dar 0,95 a um diabético custa 0,05; dar 0,126 custa 2,07. Errar com confiança sai caro.
4. Soma o **freio**: alpha ÷ (2 × nº de pacientes) × soma dos pesos². Com alpha = 3, pesos grandes ficam caros, e a rede não decora pacientes isolados.
5. O **lbfgs** ajusta pesos e vieses na direção que reduz o erro, usando todos os pacientes de uma vez. Repete até o erro parar de cair (44 a 58 passos aqui).

O lbfgs foi escolhido porque, numa base pequena, chega sempre ao mesmo resultado. Com o adam, que usa lotes sorteados, a nota ia de 0,884 a 0,912 só trocando a semente.

---

## 8. O código comentado

`src/models/neural_network.py`:

```python
# As 9 numéricas do config + as 4 colunas de corte. O que não aparece aqui
# (ordinal, categóricas, escala Robust) vem do padrão do config.py.
OPCOES_PREP = {"numericas": config.COLS_NUMERICAS + CORTES_CLINICOS}


class MLPMedia(ClassifierMixin, BaseEstimator):   # herda o formato do scikit-learn

    def __init__(self, hidden_layer_sizes=(4,), alpha=10.0, activation="relu",
                 sementes=(42, 101, 202, 303, 404), limiar=0.5):
        # Só guarda as configurações, com os mesmos nomes (regra do scikit-learn).
        # Estes são valores padrão; os usados de fato vêm da GRADE.
        self.hidden_layer_sizes = hidden_layer_sizes
        self.alpha = alpha
        self.activation = activation
        self.sementes = sementes
        self.limiar = limiar

    def fit(self, X, y):
        # X já chega tratado (24 colunas). "base" é a receita de uma rede, ainda sem treino.
        base = MLPClassifier(hidden_layer_sizes=self.hidden_layer_sizes, alpha=self.alpha,
                             activation=self.activation, solver="lbfgs",
                             max_iter=4000, max_fun=50000)   # tetos de segurança
        # Uma rede por semente: copia a receita, muda o ponto de partida e treina.
        self.redes_ = [clone(base).set_params(random_state=s).fit(X, y) for s in self.sementes]
        self.classes_ = self.redes_[0].classes_            # [0, 1]
        return self

    def predict_proba(self, X):
        # Média das chances [controle, diabetes] das 5 redes, paciente por paciente.
        return np.mean([r.predict_proba(X) for r in self.redes_], axis=0)

    def predict(self, X):
        # Coluna de diabetes > limiar → 1; senão → 0.
        return (self.predict_proba(X)[:, 1] > self.limiar).astype(int)


# Valores testados pela busca. "modelo__" aponta para a etapa "modelo" do Pipeline.
GRADE = {
    "modelo__hidden_layer_sizes": [(2,)],
    "modelo__alpha": [3.0],
    "modelo__activation": ["tanh"],
    "modelo__limiar": [0.55],
}
```

**Onde fica o tratamento dos dados:** fora da classe. O `executar` (em `evaluation.py`) monta `Pipeline([("prep", tratamento), ("modelo", MLPMedia)])`. Quando o Pipeline recebe `fit` ou `predict`, ele roda o tratamento antes e entrega à `MLPMedia` as 24 colunas prontas. Na validação cruzada, o tratamento aprende medianas e escalas só com os pacientes de treino de cada rodada, o que evita vazamento de dados.
