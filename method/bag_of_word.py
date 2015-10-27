from gensim.corpora import dictionary
from gensim.models import tfidfmodel


def sparse(vec):
    t = ""
    for id,v in vec:
        t +=str(id+1) +":"+str(v) + " "
    return t
def vectorize(ori_data,train_file_path,test_file_path):
    train_data =  ori_data['train']
    corpus = []
    for label,data in train_data:
        corpus.append(data)
    dic = dictionary.Dictionary(corpus)
    corpus = [dic.doc2bow(doc)  for doc in corpus]
    tfidf = tfidfmodel.TfidfModel(corpus)
    #training data
    fp = open(train_file_path,'w')
    for label,data in train_data:
        vec= tfidf[dic.doc2bow(data)]
        fp.write("%s %s\n"%(label,sparse(vec)))
    fp.close()
    fp = open(test_file_path,'w')
    for label,data in ori_data['test']:
        vec= tfidf[dic.doc2bow(data)]
        fp.write("%s %s\n"%(label,sparse(vec)))
    fp.close()






