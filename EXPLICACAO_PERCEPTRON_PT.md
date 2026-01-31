# Algoritmo Perceptron para PLN - Explicação

## Visão Geral
Esta implementação demonstra um **classificador perceptron** para Processamento de Linguagem Natural (PLN), especificamente para análise de sentimentos binária de avaliações de filmes.

## Como o Perceptron Funciona

### 1. **Arquitetura**
```
Camada de Entrada (Features de Texto) → Soma Ponderada → Função de Ativação → Saída (0 ou 1)
```

O perceptron é a forma mais simples de rede neural com:
- **Entrada**: Vetor de características representando texto
- **Pesos**: Parâmetros aprendidos (um por característica)
- **Viés**: Parâmetro único de deslocamento aprendido
- **Ativação**: Função degrau (0 se soma < 0, caso contrário 1)

### 2. **Pipeline de Pré-processamento de Texto**

#### Tokenização
```
"Este filme foi ótimo!" → ["este", "filme", "foi", "ótimo"]
```

#### Construção do Vocabulário
- Extrair todas as palavras dos dados de treinamento
- Manter apenas as N palavras mais frequentes (ex: 1000)
- Criar mapeamento palavra-para-índice

#### Vetorização (Bag-of-Words)
```
Texto: "ótimo filme ótimo atuação"
Vocabulário: {ótimo: 0, filme: 1, atuação: 2, ruim: 3}
Vetor: [2, 1, 1, 0]  # Contagem de cada palavra
```

### 3. **Algoritmo de Treinamento**

O perceptron usa a **Regra de Aprendizado do Perceptron**:

```
Para cada exemplo de treinamento (x, y):
  1. Fazer previsão: ŷ = sign(w·x + b)
  2. Se ŷ ≠ y (previsão está errada):
     - Atualizar: w = w + η(y - ŷ)x
     - Atualizar: b = b + η(y - ŷ)
```

Onde:
- `w` = vetor de pesos
- `x` = vetor de características de entrada
- `b` = viés
- `η` = taxa de aprendizado
- `y` = rótulo verdadeiro (0 ou 1)
- `ŷ` = rótulo previsto

### 4. **Componentes Principais**

#### Extração de Features
```python
def texto_para_vetor(self, texto):
    vetor = np.zeros(len(self.vocabulario))
    tokens = self.tokenizar(texto)
    for token in tokens:
        if token in self.vocabulario:
            idx = self.vocabulario[token]
            vetor[idx] += 1  # Contar frequência
    return vetor
```

#### Previsão
```python
def prever(self, x):
    ativacao = np.dot(x, self.pesos) + self.vies
    return 1 if ativacao >= 0 else 0
```

#### Atualização de Pesos
```python
if previsao != y[i]:
    atualizacao = self.taxa_aprendizado * (y[i] - previsao)
    self.pesos += atualizacao * X[i]
    self.vies += atualizacao
```

## Exemplo Passo a Passo

### Dados de Treinamento
```
Avaliações positivas (rótulo=1):
- "Este filme foi absolutamente maravilhoso"
- "Ótima atuação e enredo brilhante"

Avaliações negativas (rótulo=0):
- "Este filme foi terrível e entediante"
- "Filme horrível com atuação ruim"
```

### Processo de Aprendizado

**Estado inicial**: Todos os pesos = 0, viés = 0

**Iteração 1**:
- Entrada: "maravilhoso filme" → vetor: [0, 1, 0, 1, 0]
- Previsão: 0 (errado, deveria ser 1)
- Atualizar pesos para "maravilhoso" e "filme" positivamente

**Iteração 2**:
- Entrada: "terrível filme" → vetor: [1, 1, 0, 0, 0]
- Previsão: 1 (errado, deveria ser 0)
- Atualizar pesos para "terrível" e "filme" negativamente

**Após convergência**:
- Palavras positivas (maravilhoso, ótimo) têm pesos positivos
- Palavras negativas (terrível, horrível) têm pesos negativos

## Formulação Matemática

### Fronteira de Decisão
```
f(x) = sign(w₁x₁ + w₂x₂ + ... + wₙxₙ + b)
```

Onde:
- x₁, x₂, ..., xₙ são frequências de palavras
- w₁, w₂, ..., wₙ são pesos aprendidos
- b é o termo de viés

### Teorema de Convergência
O perceptron tem garantia de convergência se:
1. Os dados são **linearmente separáveis**
2. Taxa de aprendizado > 0

## Vantagens

1. **Simples**: Fácil de entender e implementar
2. **Rápido**: Tempo de treinamento O(n) por iteração
3. **Interpretável**: Pesos mostram importância das palavras
4. **Aprendizado Online**: Pode atualizar com novos dados incrementalmente

## Limitações

1. **Apenas Linear**: Não pode aprender padrões não-lineares
2. **Sem Saída Probabilística**: Apenas classificação binária
3. **Sensível a Outliers**: Pontos mal classificados influenciam fortemente o aprendizado
4. **Requer Dados Linearmente Separáveis**: Não converge caso contrário

## Melhorias & Extensões

### 1. **Melhor Extração de Features**
- TF-IDF em vez de contagens brutas
- N-gramas (bigramas, trigramas)
- Embeddings de palavras (Word2Vec, GloVe)

### 2. **Classificação Multi-classe**
- Abordagem um-contra-todos
- Múltiplos perceptrons

### 3. **Algoritmos Avançados**
- Perceptron multi-camadas (MLP)
- Máquinas de Vetores de Suporte (SVM)
- Regressão Logística (adiciona ativação sigmoide)

## Exemplo de Uso

```python
# Criar perceptron
perceptron = PerceptronTexto(taxa_aprendizado=0.1, epocas=100)

# Treinar em avaliações de filmes
textos_treino = ["Ótimo filme!", "Filme terrível", ...]
rotulos_treino = [1, 0, ...]
perceptron.treinar(textos_treino, rotulos_treino)

# Fazer previsões
texto_teste = "Este foi um filme incrível"
previsao = perceptron.prever(perceptron.texto_para_vetor(texto_teste))
# previsao = 1 (positivo)
```

## Métricas de Desempenho

Da saída do exemplo:
- **Acurácia de Treinamento**: 100% (convergiu em 3 épocas)
- **Acurácia de Teste**: 100%
- **Tamanho do Vocabulário**: 65 palavras
- **Features Mais Importantes**: "decepcionante" (negativo), "incrível" (positivo)

## Quando Usar Perceptron para PLN

**Bom para**:
- Classificação de texto binária
- Modelos baseline rápidos
- Propósitos educacionais
- Modelos simples e interpretáveis

**Melhores alternativas**:
- Regressão Logística (saídas probabilísticas)
- Naive Bayes (funciona bem para texto)
- Redes Neurais (padrões complexos)
- Transformers (estado da arte, ex: BERT)

## Conceitos-Chave para Lembrar

### 📊 Fórmula Principal
```
Saída = sign(Σ(peso_i × feature_i) + viés)
```

### 🔄 Regra de Atualização
```
peso_novo = peso_antigo + taxa × erro × entrada
```

### 🎯 Condição de Convergência
- Dados devem ser **linearmente separáveis**
- Existe um hiperplano que separa as classes

### 💡 Interpretabilidade
- **Pesos positivos** → palavras que indicam classe positiva
- **Pesos negativos** → palavras que indicam classe negativa
- **Magnitude do peso** → importância da palavra

## Fluxo de Trabalho Completo

```
1. COLETA DE DADOS
   ↓
2. PRÉ-PROCESSAMENTO (tokenização, limpeza)
   ↓
3. CONSTRUÇÃO DE VOCABULÁRIO
   ↓
4. VETORIZAÇÃO (bag-of-words)
   ↓
5. TREINAMENTO (atualização iterativa de pesos)
   ↓
6. AVALIAÇÃO (métricas de desempenho)
   ↓
7. PREVISÃO (novos textos)
```

## Comparação com Outros Algoritmos

| Algoritmo | Complexidade | Interpretabilidade | Desempenho |
|-----------|--------------|-------------------|------------|
| Perceptron | Baixa | Alta | Bom (dados lineares) |
| Naive Bayes | Baixa | Alta | Bom (texto) |
| SVM | Média | Média | Muito Bom |
| Redes Neurais | Alta | Baixa | Excelente |
| BERT/Transformers | Muito Alta | Muito Baixa | Estado da Arte |

## Dicas Práticas

### ✅ Faça
- Use normalização de texto (lowercase, remoção de pontuação)
- Remova stopwords se apropriado
- Experimente diferentes taxas de aprendizado
- Valide com dados separados de teste
- Analise os pesos para interpretar o modelo

### ❌ Evite
- Usar em dados com padrões não-lineares complexos
- Ignorar desbalanceamento de classes
- Treinar sem validação
- Usar vocabulário muito grande sem seleção de features
- Esperar resultados perfeitos em problemas complexos

## Aplicações no Mundo Real

1. **Análise de Sentimentos**: Classificar reviews, tweets, comentários
2. **Detecção de Spam**: Filtrar emails indesejados
3. **Categorização de Documentos**: Organizar artigos por tópico
4. **Moderação de Conteúdo**: Detectar conteúdo inapropriado
5. **Sistemas de Recomendação**: Classificar preferências de usuários

## Recursos Adicionais

### 📚 Para Aprender Mais
- "Pattern Recognition and Machine Learning" - Bishop
- "Introduction to Machine Learning" - Alpaydin
- Curso de Machine Learning - Andrew Ng (Coursera)
- Documentação scikit-learn

### 🛠️ Ferramentas e Bibliotecas
- **scikit-learn**: Implementação pronta de perceptron
- **NLTK/spaCy**: Pré-processamento de texto
- **pandas**: Manipulação de dados
- **matplotlib**: Visualização de resultados

## Conclusão

O perceptron fornece uma base sólida para entender como modelos de machine learning processam dados de texto. Embora o PLN moderno use abordagens mais sofisticadas, os conceitos centrais—extração de features, aprendizado de pesos e combinação linear—permanecem fundamentais para todas as arquiteturas de redes neurais.

**A simplicidade do perceptron é sua maior força educacional**: ele desmistifica o aprendizado de máquina e mostra que, com a matemática certa, podemos ensinar computadores a entender linguagem humana!

### 🎓 Próximos Passos
1. Implemente variações (taxa de aprendizado adaptativa)
2. Experimente com diferentes features (TF-IDF, n-gramas)
3. Compare com outros algoritmos
4. Aplique em seus próprios dados de texto
5. Explore redes neurais multi-camadas (MLP)

**Lembre-se**: Todo grande modelo de IA começou com fundamentos simples como o perceptron. Dominar o básico é essencial para compreender o avançado! 🚀
