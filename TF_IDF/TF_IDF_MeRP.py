import numpy as np
from collections import Counter
import itertools
import json


with open('/media/camlab1/doc_drive/IU_data/images_R2_Ori/iu_annotation_R2Gen.json', 'r') as input:
    DATA = json.load(input)

with open('/media/camlab1/doc_drive/IU_data/images_R2_Ori/vocab_TF-IDF.json', 'r') as input:  # using the vocabulary of TF-IDF construction which can be different from the report generation vocabulary
    vocab = json.load(input)


docs = []
k = []
for key in DATA["train"]:
    b = key['report'].lower().replace('.', '').replace(',', '').split()
    k.append(key['id'])
    for i in range(len(b)):
        if b[i] not in vocab:
            b[i] = "<unk>"
    b = " ".join(b)
    docs.append(b)

docs_words = [d.replace(",", "").split(" ") for d in docs]
vocab = set(itertools.chain(*docs_words))


v2i = {v: i for i, v in enumerate(vocab)}
i2v = {i: v for v, i in v2i.items()}


def safe_log(x):
    mask = x != 0
    x[mask] = np.log(x[mask])
    return x


tf_methods = {
        "log": lambda x: np.log(1+x),
        "augmented": lambda x: 0.5 + 0.5 * x / np.max(x, axis=1, keepdims=True),
        "boolean": lambda x: np.minimum(x, 1),
        "log_avg": lambda x: (1 + safe_log(x)) / (1 + safe_log(np.mean(x, axis=1, keepdims=True))),
    }
idf_methods = {
        "log": lambda x: 1 + np.log(len(docs) / (x+1)),
        "prob": lambda x: np.maximum(0, np.log((len(docs) - x) / (x+1))),
        "len_norm": lambda x: x / (np.sum(np.square(x))+1),
    }


def get_tf(method="log"):
    # term frequency: how frequent a word appears in a doc
    _tf = np.zeros((len(vocab), len(docs)), dtype=np.float64)    # [n_vocab, n_doc]
    for i, d in enumerate(docs_words):
        counter = Counter(d)
        for v in counter.keys():
            _tf[v2i[v], i] = counter[v] / counter.most_common(1)[0][1]

    weighted_tf = tf_methods.get(method, None)
    if weighted_tf is None:
        raise ValueError
    return weighted_tf(_tf)


def get_idf(method="log"):
    # inverse document frequency: low idf for a word appears in more docs, mean less important
    df = np.zeros((len(i2v), 1))
    for i in range(len(i2v)):
        d_count = 0
        for d in docs_words:
            d_count += 1 if i2v[i] in d else 0
        df[i, 0] = d_count

    idf_fn = idf_methods.get(method, None)
    if idf_fn is None:
        raise ValueError
    return idf_fn(df)


def cosine_similarity(q, _tf_idf):
    unit_q = q / np.sqrt(np.sum(np.square(q), axis=0, keepdims=True))
    unit_ds = _tf_idf / np.sqrt(np.sum(np.square(_tf_idf), axis=0, keepdims=True))
    similarity = unit_ds.T.dot(unit_q).ravel()
    return similarity


def docs_score(q, len_norm=False):
    q_words = q.replace(",", "").replace(".", "").split(" ")
    # add unknown words
    unknown_v = 0
    for v in set(q_words):
        if v not in v2i:
            v2i[v] = len(v2i)
            i2v[len(v2i)-1] = v
            unknown_v += 1
    if unknown_v > 0:
        _idf = np.concatenate((idf, np.zeros((unknown_v, 1), dtype=np.float)), axis=0)
        _tf_idf = np.concatenate((tf_idf, np.zeros((unknown_v, tf_idf.shape[1]), dtype=np.float)), axis=0)
    else:
        _idf, _tf_idf = idf, tf_idf
    counter = Counter(q_words)
    q_tf = np.zeros((len(_idf), 1), dtype=np.float)     # [n_vocab, 1]
    for v in counter.keys():
        q_tf[v2i[v], 0] = counter[v]

    q_vec = q_tf * _idf            # [n_vocab, 1]

    q_scores = cosine_similarity(q_vec, _tf_idf)
    if len_norm:
        len_docs = [len(d) for d in docs_words]
        q_scores = q_scores / np.array(len_docs)
    return q_scores, q_vec


def get_keywords(n=3):
    for c in range(5):
        col = tf_idf[:, c]
        idx = np.argsort(col)[-n:]
        print("doc{}, top{} keywords {}".format(c, n, [i2v[i] for i in idx]))


tf = get_tf()           # [n_vocab, n_doc]
idf = get_idf()         # [n_vocab, 1]
tf_idf = tf * idf       # [n_vocab, n_doc]

dic = {}
i = 0
for key in k:
    dic[key] = tf_idf[:, i].tolist()
    i = i + 1

diction = {}
for key in DATA["train"]:
    id = key['id']
    diction[id] = dic[id]

with open('TF_IDF_Report.json', 'w') as fp:
     json.dump(diction, fp)

print(len(diction))

print("tf shape(vecb in each docs): ", tf.shape)
print("\ntf samples:\n", tf[:2])
print("\nidf shape(vecb in all docs): ", idf.shape)
print("\nidf samples:\n", idf[:2])
print("\ntf_idf shape: ", tf_idf.shape)
print("\ntf_idf sample:\n", tf_idf[:2])
get_keywords()
q = "there has been interval removal of right-sided central venous catheter . enteric tube is again noted coursing below the diaphragm the tip of which is seen projecting over the expected location of the body of the stomach . the cardiomediastinal silhouette is normal unchanged from prior . low lung volumes causing streaky bibasilar atelectasis and bronchovascular crowding . there is no pneumothorax . the visualized bony structures reveal no acute abnormalities . lateral view reveals mild degenerative changes of the thoracic spine ."
scores, q_tf_IDF = docs_score(q)
print(len(q_tf_IDF))
d_ids = scores.argsort()[-3:][::-1]
print("\ntop 3 docs for '{}':\n{}".format(q, [docs[i] for i in d_ids]))

