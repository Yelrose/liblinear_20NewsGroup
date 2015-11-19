import sys
import numpy as np
sys.path.append('/home/hzhengj/Workspace/wordmatrix/')
import word2mat
vector_size = 400
def sparse(vec):
	t = ""
	for id,v in enumerate(vec):
		t += str(id+1) + ":" + str(v) + " "
	return t

def make_data(data,doc,path,model):
	fp = open(path,'w')
	countf =0
	for label,sentence in data:
		vec = np.zeros(vector_size,dtype="float32")
		nword = 0
		for wd in sentence:
			if wd in model.vocab:
				word = model.vocab[wd]
				vec2 = model.syn0[word.index].reshape(model.vector_size,model.topic_size).dot(doc[countf])
				nword += 1
				vec += vec2
		if nword > 0:
			vec /= nword
		fp.write("%s %s\n" %(label,sparse(vec)))
		countf += 1
	fp.close()


class Doc:
    def __init__(self,topic,phase,length,model):
        self.topic = topic
        self.length = length
        self.phase = phase
        self.model = model
    def __getitem__(self,key):
        #vec = np.abs(np.random.randn(self.topic))
        #vec /= np.sum(vec)
        #vec = np.array(vec,dtype='float32')
        #vec = np.zeros(self.topic,dtype='float32')
        #vec[0] = 0.5
        #vec[1] = 0.5

        #return vec
        if self.phase == 'test': key += self.length

        return self.model[key]



def vectorize(ori_data,train_file_path,test_file_path):
	ntopics = 80
	doc =  np.loadtxt('tmp/model-final.theta',dtype="float32")
	train_data =  ori_data['train']
	#print ori_data['train']
	test_data = ori_data['test']
	corpus = []
	for label,data in train_data:
		corpus.append(data)
	#print corpus
	model  = word2mat.Word2Mat(corpus,sentences_vector=Doc(ntopics,'train',len(train_data),doc),topic=ntopics,size=vector_size,iter=10,min_count=5,workers=1)
	#model = word2vec.Word2Vec(corpus,size=vector_size,min_count=5)
	make_data(train_data,Doc(ntopics,'train',len(train_data),doc),train_file_path,model)
	make_data(test_data,Doc(ntopics,'test',len(train_data),doc),test_file_path,model)
