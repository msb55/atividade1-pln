import nltk
from nltk import word_tokenize
from nltk.corpus import reuters
import script
from script import text_processing, get_quantidade_documentos
# para numeros muito pequenos
from decimal import *

categories = ["acq","corn","crude","earn","grain","interest","money-fx","trade","ship","wheat"]

def count_words():
	bow = text_processing()
	
	result = []	
	for b in bow:
		d = {}
		# ocorrencia das palavras no BoW de treinamento 
		for x in b["treinamento"]:
			if x in d:
				d[x] += 1
			else:
				d[x] = 1

		result.append({"categoria": b["categoria"], "words": d, "qtde_tipos": len(d), "qtde_tokens": len(b["treinamento"]), "qtde_doc": b["qtde_doc"]})
	
	# estrutura de retorno com a categoria, o conjunto de palavras com a ocorrencia, 
	# 	a quantidade de tipos, a quantidade de tokens e a quantidade de documentos da categoria
	return result

# text: texto de entrada para o classificador
# count: saida do metodo count_words()
def naive_bayes(text,count):
	result = []

	# quantidade de documentos de todas as categorias
	qtde_all_docs = get_quantidade_documentos()
	for c in count:
		# probabilidade para a categoria c
		prob_c = 1
		for word in text:
			# caso ja tenha sido "visto" a palavra na categoria
			if word in c["words"]:
				# P(w | c) = (count(w,c) + 1) / ((+)count(w,c) + |V|)
				prob_c *= Decimal(c["words"][word] + 1) / Decimal(c["qtde_tokens"] + c["qtde_tipos"])
			else:
				# suavização Add-1 para os casos da palavra não ter sido "vista" anteriormente na categoria
				prob_c *= Decimal(1) / Decimal(c["qtde_tokens"] + c["qtde_tipos"])				
		
		# probabilidade a priori P(c) = Nc / N
		prob_priori = Decimal(c["qtde_doc"] / qtde_all_docs)
		result.append({"categoria": c["categoria"], "probabilidade": (prob_priori * prob_c)})

	# estrutura de retorno com o nome da categoria e a probabilidade do texto (documento) de entrada pertencer a ela
	return result

# a maior probabilidade dentre as categorias
def high_category(categories_prob):
	category = ""
	maximum = Decimal(0)
	for c in categories_prob:
		if c["probabilidade"] > maximum:
			category = c["categoria"]
			maximum = c["probabilidade"]

	return category

# estrutura base para a matriz com os valores de acertos e erros (tp, fp, tn, fn)
def base_struct():
	result = {}
	# {'category_name': 0, ...}
	for c in categories:
		result[c] = 0
	return result

# calculo da precisão, recall e F1 para cada categoria, e a acuracia (total)
def medidas_avaliacao(matriz):
	somatorio_diagonal = 0
	somatorio_total = 0
	for c in categories:
		recall = matriz[c][c]
		precisao = matriz[c][c]
		somatorio_diagonal += matriz[c][c]
		somatorio = 0
		for x in categories:
			somatorio += matriz[c][x]
			somatorio_total += matriz[c][x]
		recall /= somatorio
		somatorio = 0
		for x in categories:
			somatorio += matriz[x][c]
		precisao /= somatorio
		print("classe: ",c)
		print("recall: ", recall)
		print("precisao: ", precisao)
		print("F1: ", (2*recall*precisao)/(recall+precisao))	

	print("acurácia: ", (somatorio_diagonal/somatorio_total))

def main():
	count = count_words()

	# Stemming
	porter = nltk.PorterStemmer()

	# Lematização
	# wnl = nltk.WordNetLemmatizer()

	matriz = {}

	for c in categories:
		category_docs = reuters.fileids(c)

		# test set
		test_docs = list(filter(lambda doc: doc.startswith("test"),category_docs))

		struct = base_struct()
		
		for t in test_docs:
			raw = reuters.raw(t)
			porter = nltk.PorterStemmer()
			tokens = word_tokenize(raw)

			# com lematização
			# lemma = [wnl.lemmatize(t) for t in tokens]
			# documento do conjunto de testes para o naive bayes
			# result = naive_bayes(lemma,count)

			# com stemming
			stemmer = [porter.stem(t)for t in tokens]
			# documento do conjunto de testes para o naive bayes
			result = naive_bayes(stemmer,count)
			
			# qual a categoria com maior probabilidade
			res = high_category(result)

			# contabiliza na posicao da categoria selecionada
			struct[res] += 1
			
		# para a categoria c, a estrutura base contabilizada nos selecionados
		matriz[c] = struct
	
	# calcula as avaliações
	medidas_avaliacao(matriz)

main()