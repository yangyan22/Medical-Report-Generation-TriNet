import pickle
from collections import Counter
import json
import re


class JsonReader(object):
    def __init__(self, json_file):
        self.data = self.__read_json(json_file)
        self.keys = list(self.data.keys())

    def __read_json(self, filename):
        with open(filename, 'r') as f:
            data = json.load(f)
        return data

    def __getitem__(self, item):
        # return self.data[item]
          return self.data[self.keys[item]]

    def __len__(self):
        return len(self.data)


class Vocabulary(object):
    def __init__(self):
        self.word2idx = {}
        self.id2word = {}
        self.idx = 0
        self.add_word('<pad>')  # 0
        self.add_word('<start>')  # 1
        self.add_word('<end>')  # 2
        self.add_word('<unk>')  # 3

    def add_word(self, word):
        if word not in self.word2idx:
            print(word)
            self.word2idx[word] = self.idx
            self.id2word[self.idx] = word
            self.idx += 1

    def get_word_by_id(self, id):
        # print(self.id2word[id])
        return self.id2word[id]

    def __call__(self, word):
        if word not in self.word2idx:
            # print(word)  # 句子没有分
            return self.word2idx['<unk>']
        return self.word2idx[word]

    def __len__(self):  # 996
        # print(self.word2idx)  # 字典 word到id
        # print(self.id2word)  # 字典 id到word
        return len(self.word2idx)


def build_vocab(json_file, threshold):
    with open(json_file, 'r') as f:
        data = json.load(f)
    caption_reader = data["train"]
    counter = Counter()

    for item in caption_reader:
        items = item['report']
        report_cleaner = lambda t:t.replace('..', '.').replace('..', '.').replace('..', '.').replace('1. ', '') \
            .replace('. 2. ', '. ').replace('. 3. ', '. ').replace('. 4. ', '. ').replace('. 5. ', '. ') \
            .replace(' 2. ', '. ').replace(' 3. ', '. ').replace(' 4. ', '. ').replace(' 5. ', '. ') \
            .strip().lower().split('. ')
        sent_cleaner = lambda t: re.sub('[.,?;*!%^&_+():-\[\]{}]', '', t.replace('"', '').replace('/', '').
                                        replace('\\', '').replace("'", '').strip().lower())
        tokens = [sent_cleaner(sent) for sent in report_cleaner(items) if sent_cleaner(sent) != []]
        report = ' . '.join(tokens) + ' .'
        y = report.lower().split(' ')
        counter.update(y)

    words = [word for word, cnt in counter.items() if cnt > threshold and word != '']

    vocab = Vocabulary()
    json_vocab = []
    for word in words:
        json_vocab.append(word)
        vocab.add_word(word)
    length = len(json_vocab)
    print(length)
    T = 799 - length
    words2 = [wor for wor, cnt in counter.items() if cnt == 3 and wor != '']
    for word in words2:
        if T > 0:
            json_vocab.append(word)
            vocab.add_word(word)
        T = T-1
    return vocab, json_vocab


def main(json_file, threshold, vocab_path):

    vocab, json_vocab = build_vocab(json_file=json_file, threshold=threshold)
    with open(vocab_path, 'wb') as f:
        pickle.dump(vocab, f)
    # js = './vocab.json'
    # with open(js, 'w') as ff:
    #     json.dump(json_vocab, ff)
    print("Total vocabulary size:{}".format(len(vocab)))
    print("Saved path in {}".format(vocab_path))
    # with open(vocab_path, 'rb') as f:
    #     data = pickle.load(f)
    # print(data('.'))  # word2id  14


if __name__ == '__main__':
    main(json_file='/media/camlab1/doc_drive/IU_data/images_R2_Ori/iu_annotation_R2Gen.json', threshold=3, vocab_path='./vocab.pkl')
    f = open('./vocab.json', 'r')
    t = json.load(f)
    t.sort()
