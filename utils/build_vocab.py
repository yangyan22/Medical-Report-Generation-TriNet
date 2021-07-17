import pickle
from collections import Counter
import json


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
            # print(word)  
            return self.word2idx['<unk>']
        return self.word2idx[word]

    def __len__(self): 
        # print(self.word2idx)  
        # print(self.id2word)  
        return len(self.word2idx)


def build_vocab(json_file, threshold):
    caption_reader = JsonReader(json_file)
    counter = Counter()

    for items in caption_reader:
        text = items.replace('.', '').replace(',', '').replace(':', '').replace('(','').replace(')','').replace(' /', '').replace('/ ', '')\
            .replace('fifth', '').replace('  ', '')
       
        y = text.lower().split(' ')
        counter.update(y)
    # print(counter.items())  

    words = [word for word, cnt in counter.items() if cnt > threshold and word != '']
    # print(words)
    vocab = Vocabulary()

    for word in words:

        vocab.add_word(word)
    return vocab


def main(json_file, threshold, vocab_path):
    vocab = build_vocab(json_file=json_file, threshold=threshold)
    with open(vocab_path, 'wb') as f:
        pickle.dump(vocab, f)
    print("Total vocabulary size:{}".format(len(vocab)))
    print("Saved path in {}".format(vocab_path))

    with open(vocab_path, 'rb') as f:
        data = pickle.load(f)
    print(data('left'))  # word2id


if __name__ == '__main__':
    main(json_file='./report.json', threshold=2, vocab_path='../data/vocab.pkl')