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




def get_news_from_cache(path):
    fp = open(path)
    data = []
    while True:
        line = fp.readline()
        if not line: break
        line = line.strip().split('\x01')
        label =line[0]
        text = line[1:]
        data.append((label,text))
    return data


def dump_news_to_cache(data,path):
    fp = open(path,'w')
    for label,text in data:
        fp.write("%s\x01"%label)
        fp.write("\x01".join(text))
        fp.write("\n")
    fp.close()



if __name__ == '__main__':
    data = {}
    if os.path.exists('./tmp/test_cache'):
        data['test'] = get_news_from_cache('./tmp/test_cache')
    else :
        data['test'] = get_news('./20news-bydate-test')
        dump_news_to_cache(data['test'],'./tmp/test_cache')
    if os.path.exists('./tmp/train_cache'):
        data['train'] = get_news_from_cache('./tmp/train_cache')
    else :
        data['train'] = get_news('./20news-bydate-train')
        dump_news_to_cache(data['train'],'./tmp/train_cache')
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

