# Detecção de Fraude em Cartão de Crédito

Projeto de Machine Learning para detecção de fraude em transações reais de cartão de crédito. O foco está em **escolher a métrica certa** para um problema extremamente desbalanceado e **explicar o modelo** com SHAP.

## 📌 O problema

O dataset possui transações classificadas em normais (`Class = 0`) e fraudulentas (`Class = 1`). A distribuição real é:

| Classe | Total | Proporção |
|---|---:|---:|
| Normal (0) | 284.315 | 99,827% |
| Fraude (1) | 492 | 0,173% |

Com um desbalanceamento dessa ordem, **acurácia engana**. Um modelo que responde "não é fraude" para tudo acerta ~99,8% e não detecta nenhuma fraude. Por isso as métricas que guiam este projeto são:

- **Recall da fraude** — de todas as fraudes reais, quantas o modelo pegou?
- **Precisão da fraude** — das que o modelo marcou como fraude, quantas eram fraude mesmo?
- **F1 da fraude** — equilíbrio entre recall e precisão
- **PR-AUC (Average Precision)** — mais informativa que ROC-AUC em dados muito desbalanceados
- **Limiar de decisão** — em vez de usar 0,5 fixo, foi escolhido com base na curva precision-recall

## 📂 Fonte dos dados

O link original da Aula 1 aponta para o arquivo do notebook, não para o dataset. Para garantir reprodutibilidade, o CSV foi carregado diretamente no notebook através de um espelho público no Hugging Face:

```
https://huggingface.co/datasets/David-Egea/Creditcard-fraud-detection/resolve/main/creditcard.csv
```

O arquivo **não** foi incluído no repositório.

## 🛠️ Preparação dos dados

1. **Carregamento** via `pandas.read_csv` a partir do link
2. **Verificação**: 0 valores nulos; 1.081 duplicatas exatas
3. **Remoção de duplicatas**: 284.807 → 283.726 linhas
4. **Feature engineering**:
   - `log_amount = np.log1p(Amount)` — suaviza a distribuição assimétrica de valores
   - `hour = (Time // 3600) % 24` — captura padrão temporal das transações
5. **Split estratificado** (70/30) com `stratify=y` para manter a proporção de fraudes em treino e teste:
   - Treino: 331 fraudes de 198.608
   - Teste: 142 fraudes de 85.118
6. **Padronização**: `StandardScaler` dentro de um `Pipeline` apenas para a Regressão Logística (árvores não precisam de escala)
7. **Tratamento do desbalanceamento**:
   - Regressão Logística e Random Forest: `class_weight='balanced'`
   - XGBoost: `scale_pos_weight = 599,02` (razão entre negativos e positivos no treino)

## 🤖 Modelos comparados

| Modelo | Recall fraude | Precisão fraude | F1 fraude | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|---:|
| Regressão Logística | 0,8873 | 0,0523 | 0,0987 | 0,9676 | 0,6982 |
| Random Forest | 0,7254 | 0,9626 | 0,8273 | 0,9519 | **0,8272** |
| XGBoost (limiar 0,5) | 0,7746 | 0,9244 | 0,8429 | **0,9721** | 0,8219 |
| **XGBoost (limiar 0,57)** | **0,7746** | **0,9322** | **0,8462** | **0,9721** | 0,8219 |

### Leitura dos resultados

- A **Regressão Logística** atinge o maior recall (0,887), mas com precisão péssima (0,052): ela marca muitas transações normais como fraude. Isso ilustra bem por que recall sozinho não basta.
- O **Random Forest** é o mais conservador: alta precisão (0,963) e recall menor (0,725). A PR-AUC é a maior entre os modelos com limiar 0,5.
- O **XGBoost** apresentou o melhor equilíbrio entre recall e precisão, com a maior ROC-AUC. Ajustando o limiar, o F1 subiu ainda mais.

## 🎯 Limiar de decisão escolhido

O limiar padrão de 0,5 não é necessariamente o melhor em fraude. A curva precision-recall do XGBoost foi analisada e o limiar que **maximiza o F1 da classe fraude** foi selecionado:

```
Melhor limiar (F1): 0.5690
F1 nesse limiar   : 0.8462
Recall            : 0.7746
Precisão          : 0.9322
```

**Por que 0,57?** Ao subir levemente o limiar acima de 0,5, a precisão aumentou de 0,9244 para 0,9322 sem sacrificar o recall (que se manteve em 0,7746). Em um cenário real de fraude, isso significa **menos alarmes falsos** sem perder nenhuma fraude adicional — um trade-off desejável para reduzir o custo operacional de analistas revisando transações.

## 🔍 Explicabilidade com SHAP

Foi utilizado `shap.TreeExplainer` sobre o XGBoost treinado.

### Importância global (top 5 pelo XGBoost)

| Variável | Importância (XGB) | Importância (RF) |
|---|---:|---:|
| **V14** | 0,526 | 0,186 |
| V4 | 0,070 | 0,090 |
| V10 | 0,051 | 0,127 |
| V12 | 0,034 | 0,110 |
| V8 | 0,029 | 0,008 |

V14 domina as decisões do modelo. Como as variáveis V1–V28 vêm de PCA, não é possível traduzir diretamente o significado de negócio, mas é possível interpretar **direção e magnitude** do impacto.

### Resumo SHAP (summary plot)

O gráfico de resumo confirma V14 como a variável mais influente, seguida por V4, V10, V12 e V8. Para V14:

- Valores **baixos** empurram fortemente a predição para **fraude**
- Valores **altos** empurram para **normal**

Esse padrão é consistente com o que se observa em versões públicas deste dataset, onde V14 é consistentemente a feature mais discriminativa.

### Explicação individual (force plot)

Para uma transação específica classificada como fraude, o SHAP mostrou:

- **V14 e V4** empurraram a decisão **a favor da fraude** (maior peso para V14)
- **V10, Time e V12** empurraram **contra** a classificação como fraude
- O resultado final ficou acima do limiar ajustado, marcando a transação como suspeita

Isso mostra que o modelo não decide por uma única variável, mas pela combinação delas — e o SHAP permite auditar caso a caso.

## 📊 Visualizações geradas

O notebook contém, com saídas salvas:

- Distribuição das classes (countplot)
- Boxplot de `Amount` por classe (escala log)
- Histograma de `Time` por classe
- Curva ROC comparando os três modelos
- Curva Precision-Recall comparando os três modelos
- Barplot de importância das 15 variáveis mais relevantes (RF vs XGBoost)
- SHAP summary plot
- SHAP force plot de uma fraude individual

## 🚀 O que mudou em relação à Expert

- Criação da variável **`hour`** a partir de `Time`, para capturar padrão temporal
- **Ajuste do limiar de decisão** do XGBoost via curva precision-recall, escolhendo o limiar que maximiza o F1 da classe fraude (0,57 em vez de 0,5)
- Comparação da **PR-AUC** (Average Precision) além da ROC-AUC, por ser mais adequada a dados desbalanceados
- Análise explícita do trade-off **recall × precisão** em cada modelo, mostrando que a Regressão Logística com `class_weight='balanced'` atinge recall alto mas precisão insustentável

## 🔮 Evoluções possíveis

- Testar **SMOTE** e **undersampling** no treino e comparar recall com `scale_pos_weight`
- Aplicar `GridSearchCV` ou `RandomizedSearchCV` para ajuste de hiperparâmetros do XGBoost
- Criar features de comportamento ao longo do tempo (transações em sequência, frequência por janela)
- Testar outros classificadores (LightGBM, CatBoost) e comparar pelo recall

## ▶️ Como rodar

1. Abra o notebook no [Google Colab](https://colab.research.google.com/)
2. Rode a primeira célula para instalar o SHAP
3. Execute todas as células em ordem — o dataset é carregado automaticamente pelo link
4. Nenhuma credencial, upload manual ou download externo é necessário

### Bibliotecas utilizadas

- `pandas`, `numpy`
- `scikit-learn`
- `xgboost`
- `shap`
- `matplotlib`, `seaborn`

## ⚠️ Observações

- O arquivo `creditcard.csv` **não está** no repositório — é carregado em tempo de execução
- As saídas das células estão salvas no notebook (tabelas, métricas e curvas), servindo como evidência de execução
- As variáveis V1–V28 vêm de PCA e não têm interpretação de negócio direta; a análise SHAP foca na **direção e magnitude do impacto**
