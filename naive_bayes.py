import nltk
from nltk import word_tokenize
from nltk.corpus import reuters
import script
from script import text_processing

categories = ["acq","corn","crude","earn","grain","interest","money-fx","trade","veg-oil","wheat"]

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
	#print(text)
	
	result = []
	for c in count:
		prob_c = 1
		for word in text:
			if word in c["words"]:
				prob_c *= (c["words"][word] + 1) / (c["qtde"] + c["qtde_tokens"])
			else:
				prob_c *= 1 / (c["qtde"] + c["qtde_tokens"])
				
		
		result.append({"categoria": c["categoria"], "probabilidade": prob_c})

	print(result)


def main():
	count = count_words()
	porter = nltk.PorterStemmer()
	categories = ["corn"]

	for c in categories:
		category_docs = reuters.fileids(c)
		test_docs = list(filter(lambda doc: doc.startswith("test"),category_docs))
		
		stemmer = []
		main_test = []
		
		raw = reuters.raw(test_docs[0])
		porter = nltk.PorterStemmer()
		tokens = word_tokenize(raw)
		stemmer = [porter.stem(t)for t in tokens]
		naive_bayes(stemmer,count)

	


main()