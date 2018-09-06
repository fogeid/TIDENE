import nltk
import pandas as pd
from nltk.stem.porter import *
from gensim.test.utils import common_texts, get_tmpfile
from gensim.models import Word2Vec
from nltk.stem.wordnet import WordNetLemmatizer

auxtrein = []
basetrein = []

auxteste = []
baseteste = []

base_treinamento = pd.read_csv("train_wipo_alpha_300wordsFieldTxt")
base_teste = pd.read_csv("test_wipo_alpha_300wordsFieldTxt")

section = base_treinamento.section
data = base_treinamento.data

for linha in section:
        auxtrein.append(linha)

i = 0
for linha in data:
        basetrein.append((linha, auxtrein[i]))
        i += 1

section = base_teste.section
data = base_teste.data

for linha in section:
        auxteste.append(linha)

i = 0
for linha in data:
        baseteste.append((linha, auxteste[i]))
        i += 1


stopwords = nltk.corpus.stopwords.words('english')

#Função para remover as stopwords.
def removerstopwords(texto):
    frases = []
    for(palavras, emocao) in texto:
        semstop = [p for p in palavras.split() if p not in stopwords]
        frases.append((semstop, emocao))
    return frases

removerstopwords(basetrein)
removerstopwords(baseteste)

#Função para remover as stopwords e stemmer da base.
def aplicastemmer(texto):
    stemmer = nltk.stem.RSLPStemmer()
    frasesstemming = []
    for(palavras, emocao) in texto:
        comstemming = [str(stemmer.stem(p)) for p in palavras.split() if p not in stopwords]
        frasesstemming.append((comstemming, emocao))
    return frasesstemming

frasescomstemmingtreinamento = aplicastemmer(basetrein)
frasescomstemmingteste = aplicastemmer(baseteste)
#print(frasescomstemming)

#Função para listar todas as palavras da base.
def buscapalavras(frases):
    todaspalavras = []
    for(palavras, emocao) in frases:
        todaspalavras.extend(palavras)
    return todaspalavras

palavrastreinamento = buscapalavras(frasescomstemmingtreinamento)
palaavrasteste = buscapalavras(frasescomstemmingteste)
#print(palavras)

#Função para verificar a frequência de repetição das palavras.
def buscafrequencia(palavras):
    palavras = nltk.FreqDist(palavras)
    return palavras

frequenciatreinamento = buscafrequencia(palavrastreinamento)
frequenciateste = buscafrequencia(palaavrasteste)
#print(frequencia.most_common(50))

#Função para retirar as palavras repetidas.
def buscapalavrasunicas(frequencia):
    freq = frequencia.keys()
    return freq

palavrasunicastreinamento = buscapalavrasunicas(frequenciatreinamento)
palavrasunicasteste = buscapalavrasunicas(frequenciateste)
#print(palavrasunicas)

#Função que classifica se tem ou não a palavra em uma determinada frase.
def extratorpalavras(documento):
    doc = set(documento)
    caracteristicas = {}
    for palavras in palavrasunicastreinamento:
        caracteristicas['%s' %palavras] = (palavras in doc)
    return caracteristicas

caracteristicasfrase = extratorpalavras(['am', 'nov', 'dia'])
#print(caracteristicasfrase)

basecompletatreinamento = nltk.classify.apply_features(extratorpalavras, frasescomstemmingtreinamento)
basecompletateste = nltk.classify.apply_features(extratorpalavras, frasescomstemmingteste)
#print(basecompleta)

#Constrói a tabela de probabilidade usando o método Naive Bayes.
classificador = nltk.NaiveBayesClassifier.train(basecompletatreinamento)
print(classificador.labels())
print(classificador.show_most_informative_features(20))

print("Accuracy: ", nltk.classify.accuracy(classificador, basecompletateste))

erros = []
for (frase, classe) in basecompletateste:
    resultado = classificador.classify(frase)
    if(resultado != classe):
        erros.append((classe, resultado, frase))
#for (classe, resultado, frase) in erros:
#    print(classe, resultado, frase)

from nltk.metrics import ConfusionMatrix
esperado = []
previsto = []
for (frase, classe) in basecompletateste:
    resultado = classificador.classify(frase)
    previsto.append(resultado)
    esperado.append(classe)

matriz = ConfusionMatrix(esperado, previsto)
print(matriz)

#Implementação do W2V
path = get_tmpfile("word2vec.modelo")
modelo = Word2Vec(common_texts, size=100, window=5, min_count=1, workers=4)
modelo.save("word2vec.modelo")

modelo = Word2Vec.load("word2vec.modelo")
modelo.train([["hello", "world"]], total_examples=1, epochs=1)
(0, 2)
modelo = Word2Vec(sentenca, min_count=5)  # Ignora palavras que aparecem menos de 5x
vector = modelo.wv['computer']  # vetor numpy de uma palavra

