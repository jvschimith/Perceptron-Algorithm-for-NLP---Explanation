"""
Algoritmo Perceptron para Classificação de Texto em PLN
Este exemplo demonstra análise de sentimentos em avaliações de filmes
"""

import numpy as np
from collections import Counter
import re


class PerceptronTexto:
    """
    Um classificador perceptron simples para dados de texto.
    Usa classificação binária de sentimentos (positivo/negativo).
    """
    
    def __init__(self, taxa_aprendizado=0.01, epocas=100):
        self.taxa_aprendizado = taxa_aprendizado
        self.epocas = epocas
        self.pesos = None
        self.vies = 0
        self.vocabulario = {}
        
    def tokenizar(self, texto):
        """Tokenização simples: minúsculas e divisão em não-alfanuméricos"""
        texto = texto.lower()
        tokens = re.findall(r'\b\w+\b', texto)
        return tokens
    
    def construir_vocabulario(self, textos, max_features=1000):
        """Construir vocabulário a partir dos textos de treino"""
        todos_tokens = []
        for texto in textos:
            todos_tokens.extend(self.tokenizar(texto))
        
        # Obter palavras mais comuns
        contagem_palavras = Counter(todos_tokens)
        mais_comuns = contagem_palavras.most_common(max_features)
        
        # Criar mapeamento palavra para índice
        self.vocabulario = {palavra: idx for idx, (palavra, _) in enumerate(mais_comuns)}
        print(f"Tamanho do vocabulário: {len(self.vocabulario)}")
        
    def texto_para_vetor(self, texto):
        """Converter texto em vetor de características usando bag-of-words"""
        vetor = np.zeros(len(self.vocabulario))
        tokens = self.tokenizar(texto)
        
        for token in tokens:
            if token in self.vocabulario:
                idx = self.vocabulario[token]
                vetor[idx] += 1  # Contar frequência
        
        return vetor
    
    def prever(self, x):
        """Fazer previsão usando os pesos atuais"""
        ativacao = np.dot(x, self.pesos) + self.vies
        return 1 if ativacao >= 0 else 0
    
    def treinar(self, textos, rotulos):
        """
        Treinar o perceptron
        textos: lista de strings de texto
        rotulos: lista de rótulos binários (0 ou 1)
        """
        # Construir vocabulário a partir dos dados de treino
        self.construir_vocabulario(textos)
        
        # Converter textos em vetores
        X = np.array([self.texto_para_vetor(texto) for texto in textos])
        y = np.array(rotulos)
        
        # Inicializar pesos
        n_features = X.shape[1]
        self.pesos = np.zeros(n_features)
        self.vies = 0
        
        # Loop de treinamento
        for epoca in range(self.epocas):
            erros = 0
            
            for i in range(len(X)):
                # Fazer previsão
                previsao = self.prever(X[i])
                
                # Atualizar pesos se a previsão estiver errada
                if previsao != y[i]:
                    atualizacao = self.taxa_aprendizado * (y[i] - previsao)
                    self.pesos += atualizacao * X[i]
                    self.vies += atualizacao
                    erros += 1
            
            if epoca % 10 == 0:
                acuracia = (len(X) - erros) / len(X) * 100
                print(f"Época {epoca}: Acurácia = {acuracia:.2f}%")
            
            # Parada antecipada se acurácia perfeita
            if erros == 0:
                print(f"Convergiu na época {epoca}")
                break
    
    def avaliar(self, textos, rotulos):
        """Avaliar modelo nos dados de teste"""
        X = np.array([self.texto_para_vetor(texto) for texto in textos])
        y = np.array(rotulos)
        
        previsoes = [self.prever(x) for x in X]
        acuracia = np.mean(previsoes == y) * 100
        
        return acuracia, previsoes
    
    def obter_top_features(self, n=10):
        """Obter top features ponderadas (palavras mais indicativas)"""
        if self.pesos is None:
            return []
        
        pesos_palavras = [(palavra, self.pesos[idx]) 
                         for palavra, idx in self.vocabulario.items()]
        
        # Ordenar por peso absoluto
        pesos_palavras.sort(key=lambda x: abs(x[1]), reverse=True)
        
        return pesos_palavras[:n]


def main():
    """Demonstração do perceptron em dados de avaliações de filmes"""
    
    # Dados de treinamento de amostra: avaliações de filmes com rótulos de sentimento
    # Rótulo: 1 = positivo, 0 = negativo
    textos_treino = [
        "Este filme foi absolutamente maravilhoso e incrível",
        "Eu amei cada minuto deste filme foi fantástico",
        "Ótima atuação e enredo brilhante altamente recomendado",
        "Melhor filme que vi em anos verdadeiramente excepcional",
        "Performances incríveis e cinematografia linda",
        "Este filme foi terrível e entediante perda de tempo",
        "Filme horrível com atuação ruim e enredo fraco",
        "Eu odiei este filme foi completamente decepcionante",
        "Pior filme já feito experiência muito ruim",
        "Enredo horrível e direção terrível não recomendado",
        "O filme foi excelente com performances soberbas",
        "Filme excepcional com efeitos visuais incríveis",
        "Isto foi uma obra-prima trabalho verdadeiramente brilhante",
        "Filme decepcionante com personagens e enredo fracos",
        "Entediante e previsível filme de qualidade muito ruim"
    ]
    
    rotulos_treino = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0]
    
    # Dados de teste
    textos_teste = [
        "Este foi um ótimo filme com atuação maravilhosa",
        "Filme terrível eu não recomendaria",
        "Enredo incrível e performances brilhantes",
        "Filme muito ruim com qualidade fraca"
    ]
    
    rotulos_teste = [1, 0, 1, 0]
    
    print("=" * 60)
    print("PERCEPTRON PARA ANÁLISE DE SENTIMENTOS EM PLN")
    print("=" * 60)
    print()
    
    # Criar e treinar perceptron
    print("Treinando perceptron...")
    print("-" * 60)
    perceptron = PerceptronTexto(taxa_aprendizado=0.1, epocas=100)
    perceptron.treinar(textos_treino, rotulos_treino)
    
    print()
    print("-" * 60)
    print("Treinamento completo!")
    print()
    
    # Avaliar nos dados de teste
    print("Avaliando nos dados de teste...")
    print("-" * 60)
    acuracia, previsoes = perceptron.avaliar(textos_teste, rotulos_teste)
    print(f"Acurácia no Teste: {acuracia:.2f}%")
    print()
    
    # Mostrar previsões
    print("Previsões no Teste:")
    for i, (texto, prev, real) in enumerate(zip(textos_teste, previsoes, rotulos_teste)):
        sentimento = "POSITIVO" if prev == 1 else "NEGATIVO"
        correto = "✓" if prev == real else "✗"
        print(f"{correto} Avaliação {i+1}: {sentimento}")
        print(f"  Texto: '{texto}'")
        print()
    
    # Mostrar features mais importantes
    print("-" * 60)
    print("Top 10 Palavras Mais Importantes:")
    print("-" * 60)
    top_features = perceptron.obter_top_features(10)
    for palavra, peso in top_features:
        direcao = "positivo" if peso > 0 else "negativo"
        print(f"{palavra:15s}: {peso:7.4f} ({direcao})")
    print()
    
    # Previsão interativa
    print("=" * 60)
    print("Experimente suas próprias avaliações:")
    print("=" * 60)
    avaliacoes_customizadas = [
        "Este filme foi absolutamente brilhante e divertido",
        "Perda de tempo e dinheiro muito decepcionante"
    ]
    
    for avaliacao in avaliacoes_customizadas:
        vetor = perceptron.texto_para_vetor(avaliacao)
        previsao = perceptron.prever(vetor)
        sentimento = "POSITIVO" if previsao == 1 else "NEGATIVO"
        print(f"Avaliação: '{avaliacao}'")
        print(f"Previsão: {sentimento}")
        print()


if __name__ == "__main__":
    main()
