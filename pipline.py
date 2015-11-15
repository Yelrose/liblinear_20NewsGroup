import os
import sys
import logging
import re
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
sys.path.append('method')

def preprocess(sentence):
    sentence = sentence.lower()
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(sentence)
    filtered_words = [w for w in tokens if not w in stopwords.words('english')]
    #filtered_words = filter(lambda token: token not in stopwords.words('english'))
    return " ".join(filtered_words)


def load_news_from_path(path):
    lines = open(path).readlines()
    content = []
    start_content = False
    print path
    for line in lines:
        if start_content is False:
            m = re.match(r'[a-zA-Z]+:',line)
            if m:
                continue
            else:
                start_content = True
        try:
            for wd in preprocess(line).split():
                if len(wd) <= 0: continue
                wd = wd.lower()
                try:
                    wd.decode("utf-8")
                except:
                    continue
                content.append(wd)
        except:
            continue
        content.append("<\s>")
    return content


def get_news(path):
    data = []
    for root,dirs,files in os.walk(path):
        if len(dirs) > 0:
            labels = dirs
            label2in = {}
            for num,label in enumerate(labels):
                label2in[label] =num + 1
        for file in files:
            context = load_news_from_path(root+'/' +file)
            label = label2in[root.split('/')[-1]]
            data.append((label,context))
    return data






if __name__ == '__main__':
    data = {}
    data['test'] = get_news('./20news-bydate-test')
    data['train'] = get_news('./20news-bydate-train')
    methods = open('method/method.list').readlines()
    fp = open('./tmp/run.sh','w')
    fp.write('echo \"start\" > res.txt\n')
    for method in methods:
        method = method.strip()
        moo = __import__(method)
        print 'making vectors by method ',method
        moo.vectorize(data,'tmp/'+method+'_train','tmp/'+method+'_test')
        fp.write('echo \"%s\" >> res.txt\n' % method)
        fp.write("../liblinear-2.1/train -s 2 %s_train\n" % method)
        fp.write("../liblinear-2.1/predict %s_test %s_train.model log.txt >> res.txt\n" % (method,method))
    fp.close()

