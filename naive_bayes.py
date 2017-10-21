import script
from script import text_processing

def naive_bayes():
	bow = text_processing()

	for b in bow:
		result = [(x,b.treinamento.count(x)) for x in b.treinamento]
		print(result)
	

naive_bayes()


