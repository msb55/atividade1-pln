import nltk
from nltk import word_tokenize
from nltk.corpus import reuters

nltk.download('reuters')
nltk.download('punkt')
nltk.download('wordnet')

# quantidade de documentos de todas as categorias (global)
quantidade_documentos = 0

def text_processing():
    global quantidade_documentos
    
    # Algoritmo de Porter (Stemming)
    porter = nltk.PorterStemmer()

    # Lematização
    # wnl = nltk.WordNetLemmatizer()

    # As 10 categorias com mais documentos
    categories = ["acq","corn","crude","earn","grain","interest","money-fx","trade","ship","wheat"]

    # Bag of Words
    bow = []

    for c in categories:
        # documentos de uma categoria
        category_docs = reuters.fileids(c)

        # apenas os de treinamento
        train_docs = list(filter(lambda doc: doc.startswith("train"),
                            category_docs)) 
        
        # training set
        main_train = []
        for train in train_docs:
            raw = reuters.raw(train)
            # tokemização
            tokens = word_tokenize(raw)

            # com lematização
            # lemma = [wnl.lemmatize(t) for t in tokens]
            # main_train += lemma

            # com stemming
            stemmer = [porter.stem(t) for t in tokens]
            main_train += stemmer
        
        quantidade_documentos += len(train_docs)      

        # estrutura com o nome da categoria, BoW do set de treinamento e a quantidade de documentos
        bow.append({'categoria': c, 'treinamento': main_train, 'qtde_doc': len(train_docs)})
        
    return bow

def get_quantidade_documentos():
    return quantidade_documentos
