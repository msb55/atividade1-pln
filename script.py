import nltk
from nltk import word_tokenize
from nltk.corpus import reuters

nltk.download('reuters')
nltk.download('punkt')

def text_processing():
    porter = nltk.PorterStemmer()
    categories = ["acq","corn","crude","earn","grain","interest","money-fx","trade","ship","wheat"]
    # categories = ["wheat", "ship", "corn"]
    bow = []
    # Documents in a category
    for c in categories:
        category_docs = reuters.fileids(c)

        train_docs = list(filter(lambda doc: doc.startswith("train"),
                            category_docs)) 
        
        
        # training set
        main_train = []
        for train in train_docs:
            raw = reuters.raw(train)
            tokens = word_tokenize(raw)            
            stemmer = [porter.stem(t) for t in tokens]
            main_train += stemmer
        
        # test set
      

        bow.append({'categoria': c, 'treinamento': main_train})
        
    return bow

