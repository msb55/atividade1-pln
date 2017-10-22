import nltk
from nltk import word_tokenize
from nltk.corpus import reuters
import script
from script import text_processing
from decimal import *

categories = ["acq","corn","crude","earn","grain","interest","money-fx","trade","ship","wheat"]

def count_words():
	bow = text_processing()
	
	result = []	
	for b in bow:
		d = {}
		for x in b["treinamento"]:
			if x in d:
				d[x] += 1
			else:
				d[x] = 1

		result.append({"categoria": b["categoria"], "words": d, "qtde_tokens": len(d), "qtde": len(b["treinamento"])})
	
	return result

def naive_bayes(text,count):
	result = []
	for c in count:
		prob_c = 1
		for word in text:
			if word in c["words"]:
				prob_c *= Decimal(c["words"][word] + 1) / Decimal(c["qtde"] + c["qtde_tokens"])
			else:
				prob_c *= Decimal(1) / Decimal(c["qtde"] + c["qtde_tokens"])				
		
		result.append({"categoria": c["categoria"], "probabilidade": prob_c})

	return result

def high_class(categories_prob):
	category = ""
	maximum = Decimal(0)
	for c in categories_prob:
		if c["probabilidade"] > maximum:
			category = c["categoria"]
			maximum = c["probabilidade"]

	return category

def base_struct():
	result = {}
	for c in categories:
		result[c] = 0
	return result

def main():
	count = count_words()
	porter = nltk.PorterStemmer()
	matriz = {}

	# categories = ["grain"]
	# categories = ["wheat", "ship", "corn"]

	for c in categories:
		category_docs = reuters.fileids(c)
		test_docs = list(filter(lambda doc: doc.startswith("test"),category_docs))

		struct = base_struct()
		acc = 0
		err = 0
		for t in test_docs:
			raw = reuters.raw(t)
			porter = nltk.PorterStemmer()
			tokens = word_tokenize(raw)
			stemmer = [porter.stem(t)for t in tokens]
			result = naive_bayes(stemmer,count)
			
			res = high_class(result)
			struct[res] += 1

			if c == res:
				acc += 1
				# print("ACERTOU")
			else:
				err += 1
				# print("ERRRRROU")
			
		matriz[c] = struct
		# print(acc)
		# print(err)
	
	print(matriz)

main()