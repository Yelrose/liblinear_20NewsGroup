import os
def load_news_from_path(path,stopwords):
    lines = open(path).readlines()
    content = []
    start_content = False
    for line in lines:
        line = line.strip()
        if start_content is False:
            if line[0:6] == 'Lines:': start_content = True
            continue
        for wd in line.split(' '):
            if len(wd) == 0: continue
            wd = wd.lower()
            if wd not in stopwords:
                content.append(wd)
        content.append("<\s>")
    return content


def get_news(path,stopwords):
    data = []
    for root,dirs,files in os.walk(path):
        for file in files:
            context = load_news_from_path(root+'/' +file,stopwords)
            label = root.split('/')[-1]
            data.append((label,context))
    return data






if __name__ == '__main__':
    stopwords = open('stopwords.txt').readlines()
    stopwords = [word.strip() for word in stopwords]
    data = {}
    data['test'] = get_news('./20news-bydate-test',stopwords)
    data['train'] = get_news('./20news-bydate-train',stopwords)

