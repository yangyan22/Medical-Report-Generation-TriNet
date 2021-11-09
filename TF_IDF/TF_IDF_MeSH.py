import json
import math

with open('/media/camlab1/doc_drive/IU_data/images_R2_Ori/mesh_data.json', 'r') as input:
    DATA = json.load(input)
    Y_length = len(DATA["train"])
print(DATA)
print(Y_length)

vocab = ['normal', 'degenerative change', 'granuloma', 'opacity', 'atelectasis', 'cardiomegaly', 'scar', 'pleural effusion', 'aorta', 'fracture', 'emphysema', 'pneumonia', 'sternotomy', 'diaphragm', 'nodule', 'infiltrates', 'deformity', 'osteophytes', 'copd', 'edema', 'support devices', 'eventration', 'thoracic vertebrae', 'tortuous aorta', 'cabg', 'scoliosis', 'hyperinflation', 'calcinosis', 'hiatal hernia', 'effusion']
# vocab = ['normal', 'degenerative change', 'granuloma', 'opacity', 'atelectasis', 'cardiomegaly', 'scar', 'pleural effusion', 'aorta', 'fracture', 'emphysema', 'pneumonia', 'sternotomy', 'diaphragm', 'nodule', 'infiltrates', 'deformity', 'osteophytes', 'copd', 'edema', 'support devices', 'eventration', 'thoracic vertebrae', 'tortuous aorta', 'cabg', 'scoliosis', 'hyperinflation', 'calcinosis', 'hiatal hernia', 'effusion', 'others']
IDF = [0] * len(vocab)
for key in DATA["train"]:
    b = DATA["train"][key]
    for i in range(len(vocab)):
        if vocab[i] in b:
            IDF[i] = IDF[i] + 1

diction = {"train": {}, "val": {}, "test": {}}
for key in DATA["train"]:
    TF_IDF = [0] * len(vocab)
    b = DATA["train"][key]
    X_length = len(b)
    for word in b:
        for i in range(len(vocab)):
            if word == vocab[i]:
                TF_IDF[i] = (1 / X_length) * math.log(Y_length/(IDF[i]))
    diction["train"][key] = TF_IDF


# ****************for test and val, we don't construct the TF-IDF, but record the binary class information
for key in DATA["test"]:
    TF_IDF = [0] * len(vocab)
    b = DATA["test"][key]
    for word in b:
        for i in range(len(vocab)):
            if word == vocab[i]:
                TF_IDF[i] = 1
    diction["test"][key] = TF_IDF

for key in DATA["val"]:
    TF_IDF = [0] * len(vocab)
    b = DATA["val"][key]
    for word in b:
        for i in range(len(vocab)):
            if word == vocab[i]:
                TF_IDF[i] = 1
    diction["val"][key] = TF_IDF

with open('TF_IDF_Mesh.json', 'w') as fp:
     json.dump(diction, fp)

