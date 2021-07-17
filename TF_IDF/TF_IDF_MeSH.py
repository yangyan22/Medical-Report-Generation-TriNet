import json
import math

with open('mesh_word.json', 'r') as input:  
    DATA = json.load(input)
    Y_length = len(DATA)

vocab = ['normal', 'degenerative change', 'opacity', 'granuloma', 'atelectasis', 'cardiomegaly', 'scar', 'pleural effusion', 'aorta', 'fracture', 'sternotomy', 'emphysema', 'pneumonia', 'infiltrates', 'osteophytes', 'copd', 'nodule', 'edema', 'deformity', 'diaphragm', 'thoracic vertebrae', 'cabg', 'arthritic changes', 'hiatal hernia', 'support devices', 'hyperinflation', 'tortuous aorta', 'congestion', 'hyperexpansion', 'scoliosis', 'others']

IDF = [0] * len(vocab)
for key in DATA:
    b = DATA[key]
    for i in range(len(vocab)):
        if vocab[i] in b:
            IDF[i] = IDF[i] + 1

diction = {}
for key in DATA:
    TF_IDF = [0] * len(vocab)  
    b = DATA[key]
    X_length = len(b)
    for word in b:
        for i in range(len(vocab)):
            if word == vocab[i]:
                TF_IDF[i] = (1 / X_length) * math.log(Y_length/(IDF[i]))
    diction[key] = TF_IDF

with open('TF_IDF_Mesh.json', 'w') as fp:
     json.dump(diction, fp)

