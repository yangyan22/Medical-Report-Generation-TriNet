import pickle
import json

f = open('/media/camlab1/doc_drive/IU_data/images_R2_Ori/img2othersFull.pkl', 'rb')
mesh_data = pickle.load(f)
data = {}
for i in mesh_data:
    key = i.split("_")[0]
    data[key] = mesh_data[i][-1]

with open('/media/camlab1/doc_drive/IU_data/images_R2_Ori/iu_annotation_R2Gen.json', 'r') as input:
    key_train = json.load(input)["train"]

with open('/media/camlab1/doc_drive/IU_data/images_R2_Ori/iu_annotation_R2Gen.json', 'r') as input:
    key_test = json.load(input)["test"]

with open('/media/camlab1/doc_drive/IU_data/images_R2_Ori/iu_annotation_R2Gen.json', 'r') as input:
    key_val = json.load(input)["val"]

#**************************the code for selecting the top 30 MeSH***************************
t = []
tags = []
for i in range(len(key_train)):
    key = key_train[i]['id'].split('_')[0]
    value = data[key]
    value = list(set(value))  # no repeat
    for j in range(len(value)):   # unify expression of the similar MeSH
        value[j] = value[j].replace('atelectases', 'atelectasis')
        value[j] = value[j].replace('scarring', 'scar')
        value[j] = value[j].replace('edemas', 'edema')
        value[j] = value[j].replace('emphysemas', 'emphysema')
        value[j] = value[j].replace('pulmonary atelectasis', 'atelectasis') #central venous catheters
        value[j] = value[j].replace('fractures, bone', 'fracture')
        value[j] = value[j].replace('scolioses', 'scoliosis')
        value[j] = value[j].replace('pleural effusions', 'pleural effusion')
        value[j] = value[j].replace('granulomatous disease', 'granuloma')
        value[j] = value[j].replace('pulmonary emphysema', 'emphysema')
        value[j] = value[j].replace('bilateral pleural effusion', 'pleural effusion')
        value[j] = value[j].replace('aorta, thoracic', 'aorta')
        value[j] = value[j].replace('thoracic aorta', 'aorta').replace('fractures', 'fracture')
        value[j] = value[j].replace('catheterization, central venous', 'central venous catheter')
        value[j] = value[j].replace('central venous catheters', 'central venous catheter')
        value[j] = value[j].replace('pulmonary disease, chronic obstructive', 'copd')
        value[j] = value[j].replace('rib fractures', 'rib fracture') # chronic lung disease
        value[j] = value[j].replace('chronic lung disease', 'copd')
        value[j] = value[j].replace('right-sided pleural effusion', 'pleural effusion')
        value[j] = value[j].replace('coronary artery bypass', 'cabg')
        value[j] = value[j].replace('focal atelectasis', 'atelectasis')
        value[j] = value[j].replace('chronic granuloma', 'granuloma')
        value[j] = value[j].replace('calcified granuloma', 'granuloma')
        value[j] = value[j].replace('granulomatous infection', 'granuloma')
        value[j] = value[j].replace('spinal osteophytosis', 'osteophytes')
        value[j] = value[j].replace('pulmonary edema', 'edema')
        value[j] = value[j].replace('rib fracture', 'fracture').replace('atheroscleroses', 'atherosclerosis')
        value[j] = value[j].replace('central venous catheter', 'support devices')
        value[j] = value[j].replace('pneumonitis', 'pneumonia')
        value[j] = value[j].replace('right upper lobe pneumonia', 'pneumonia')
        value[j] = value[j].replace('granulomas', 'granuloma').replace('esophagectomies', 'esophagectomy')
        value[j] = value[j].replace('copd, severe early-onset', 'copd')
        value[j] = value[j].replace('hypertension, pulmonary', 'pulmonary hypertension')
        value[j] = value[j].replace('obesity, morbid', 'obesity').replace('catheters', 'catheter').replace('fibroses', 'fibrosis')
        value[j] = value[j].replace('bronchiectases', 'bronchiectasis')
        value[j] = value[j].replace('mitral valve replacement', 'valve replacement').replace('aortic aneurysm, thoracic', 'aortic aneurysm')
        value[j] = value[j].replace('hernia, hiatal', 'hiatal hernia').replace('kyphosis', 'kyphoses')
    value = list(set(value))
    for j in range(len(value)):
        t.append(value[j])
dict = {}
for key in t:
    dict[key] = dict.get(key, 0) + 1
sorted_x = sorted(dict.items(), key=lambda x: x[1], reverse=True)

for i in range(30):
    tags.append(sorted_x[0:30][i][0])
print(tags)

#**************************the code for assigning MeSh for each study***************************
# tags = ['normal', 'degenerative change', 'granuloma', 'opacity', 'atelectasis', 'cardiomegaly', 'scar', 'pleural effusion', 'aorta', 'fracture', 'emphysema', 'pneumonia', 'sternotomy', 'diaphragm', 'nodule', 'infiltrates', 'deformity', 'osteophytes', 'copd', 'edema', 'support devices', 'eventration', 'thoracic vertebrae', 'tortuous aorta', 'cabg', 'scoliosis', 'hyperinflation', 'calcinosis', 'hiatal hernia', 'effusion', "others"]  # if others is needed.

tags = ['normal', 'degenerative change', 'granuloma', 'opacity', 'atelectasis', 'cardiomegaly', 'scar', 'pleural effusion', 'aorta', 'fracture', 'emphysema', 'pneumonia', 'sternotomy', 'diaphragm', 'nodule', 'infiltrates', 'deformity', 'osteophytes', 'copd', 'edema', 'support devices', 'eventration', 'thoracic vertebrae', 'tortuous aorta', 'cabg', 'scoliosis', 'hyperinflation', 'calcinosis', 'hiatal hernia', 'effusion']
diction = {"train": {}, "val": {}, "test": {}}
for i in range(len(key_train)):
    key = key_train[i]['id'].split('_')[0]
    value = data[key]
    value = list(set(value))
    for j in range(len(value)):
        value[j] = value[j].replace('atelectases', 'atelectasis')
        value[j] = value[j].replace('scarring', 'scar')
        value[j] = value[j].replace('edemas', 'edema')
        value[j] = value[j].replace('emphysemas', 'emphysema')
        value[j] = value[j].replace('pulmonary atelectasis', 'atelectasis')
        value[j] = value[j].replace('fractures, bone', 'fracture')
        value[j] = value[j].replace('scolioses', 'scoliosis')
        value[j] = value[j].replace('pleural effusions', 'pleural effusion')
        value[j] = value[j].replace('granulomatous disease', 'granuloma')
        value[j] = value[j].replace('pulmonary emphysema', 'emphysema')
        value[j] = value[j].replace('bilateral pleural effusion', 'pleural effusion')
        value[j] = value[j].replace('aorta, thoracic', 'aorta')
        value[j] = value[j].replace('thoracic aorta', 'aorta').replace('fractures', 'fracture')
        value[j] = value[j].replace('catheterization, central venous', 'central venous catheter')
        value[j] = value[j].replace('central venous catheters', 'central venous catheter')
        value[j] = value[j].replace('pulmonary disease, chronic obstructive', 'copd')
        value[j] = value[j].replace('rib fractures', 'rib fracture')
        value[j] = value[j].replace('chronic lung disease', 'copd')
        value[j] = value[j].replace('right-sided pleural effusion', 'pleural effusion')
        value[j] = value[j].replace('coronary artery bypass', 'cabg')
        value[j] = value[j].replace('focal atelectasis', 'atelectasis')
        value[j] = value[j].replace('chronic granuloma', 'granuloma')
        value[j] = value[j].replace('calcified granuloma', 'granuloma')
        value[j] = value[j].replace('granulomatous infection', 'granuloma')
        value[j] = value[j].replace('spinal osteophytosis', 'osteophytes')
        value[j] = value[j].replace('pulmonary edema', 'edema')
        value[j] = value[j].replace('rib fracture', 'fracture').replace('atheroscleroses', 'atherosclerosis')
        value[j] = value[j].replace('central venous catheter', 'support devices')
        value[j] = value[j].replace('pneumonitis', 'pneumonia')
        value[j] = value[j].replace('right upper lobe pneumonia', 'pneumonia')
        value[j] = value[j].replace('granulomas', 'granuloma').replace('esophagectomies', 'esophagectomy')
        value[j] = value[j].replace('copd, severe early-onset', 'copd')
        value[j] = value[j].replace('hypertension, pulmonary', 'pulmonary hypertension')
        value[j] = value[j].replace('obesity, morbid', 'obesity').replace('catheters', 'catheter').replace('fibroses', 'fibrosis')
        value[j] = value[j].replace('bronchiectases', 'bronchiectasis')
        value[j] = value[j].replace('mitral valve replacement', 'valve replacement').replace('aortic aneurysm, thoracic', 'aortic aneurysm')
        value[j] = value[j].replace('hernia, hiatal', 'hiatal hernia').replace('kyphosis', 'kyphoses')
    value = list(set(value))
    # for tag in value:  # if others is needed
    #     if tag not in tags:
    #         value.append('others')
    # value = list(set(value))

    tag_word = []
    for tag in value:
        if tag in tags:
            tag_word.append(tag)
    diction["train"][key] = tag_word

for i in range(len(key_val)):
    key = key_val[i]['id'].split('_')[0]
    value = data[key]
    value = list(set(value))
    for j in range(len(value)):
        value[j] = value[j].replace('atelectases', 'atelectasis')
        value[j] = value[j].replace('scarring', 'scar')
        value[j] = value[j].replace('edemas', 'edema')
        value[j] = value[j].replace('emphysemas', 'emphysema')
        value[j] = value[j].replace('pulmonary atelectasis', 'atelectasis')
        value[j] = value[j].replace('fractures, bone', 'fracture')
        value[j] = value[j].replace('scolioses', 'scoliosis')
        value[j] = value[j].replace('pleural effusions', 'pleural effusion')
        value[j] = value[j].replace('granulomatous disease', 'granuloma')
        value[j] = value[j].replace('pulmonary emphysema', 'emphysema')
        value[j] = value[j].replace('bilateral pleural effusion', 'pleural effusion')
        value[j] = value[j].replace('aorta, thoracic', 'aorta')
        value[j] = value[j].replace('thoracic aorta', 'aorta').replace('fractures', 'fracture')
        value[j] = value[j].replace('catheterization, central venous', 'central venous catheter')
        value[j] = value[j].replace('central venous catheters', 'central venous catheter')
        value[j] = value[j].replace('pulmonary disease, chronic obstructive', 'copd')
        value[j] = value[j].replace('rib fractures', 'rib fracture')
        value[j] = value[j].replace('chronic lung disease', 'copd')
        value[j] = value[j].replace('right-sided pleural effusion', 'pleural effusion')
        value[j] = value[j].replace('coronary artery bypass', 'cabg')
        value[j] = value[j].replace('focal atelectasis', 'atelectasis')
        value[j] = value[j].replace('chronic granuloma', 'granuloma')
        value[j] = value[j].replace('calcified granuloma', 'granuloma')
        value[j] = value[j].replace('granulomatous infection', 'granuloma')
        value[j] = value[j].replace('spinal osteophytosis', 'osteophytes')
        value[j] = value[j].replace('pulmonary edema', 'edema')
        value[j] = value[j].replace('rib fracture', 'fracture').replace('atheroscleroses', 'atherosclerosis')
        value[j] = value[j].replace('central venous catheter', 'support devices')
        value[j] = value[j].replace('pneumonitis', 'pneumonia')
        value[j] = value[j].replace('right upper lobe pneumonia', 'pneumonia')
        value[j] = value[j].replace('granulomas', 'granuloma').replace('esophagectomies', 'esophagectomy')
        value[j] = value[j].replace('copd, severe early-onset', 'copd')
        value[j] = value[j].replace('hypertension, pulmonary', 'pulmonary hypertension')
        value[j] = value[j].replace('obesity, morbid', 'obesity').replace('catheters', 'catheter').replace('fibroses', 'fibrosis')
        value[j] = value[j].replace('bronchiectases', 'bronchiectasis')
        value[j] = value[j].replace('mitral valve replacement', 'valve replacement').replace('aortic aneurysm, thoracic', 'aortic aneurysm')
        value[j] = value[j].replace('hernia, hiatal', 'hiatal hernia').replace('kyphosis', 'kyphoses')
    value = list(set(value))
    # for tag in value:  # if others is needed
    #     if tag not in tags:
    #         value.append('others')
    # value = list(set(value))

    tag_word = []
    for tag in value:
        if tag in tags:
            tag_word.append(tag)
    diction["val"][key] = tag_word


for i in range(len(key_test)):
    key = key_test[i]['id'].split('_')[0]
    value = data[key]
    value = list(set(value))
    for j in range(len(value)):
        value[j] = value[j].replace('atelectases', 'atelectasis')
        value[j] = value[j].replace('scarring', 'scar')
        value[j] = value[j].replace('edemas', 'edema')
        value[j] = value[j].replace('emphysemas', 'emphysema')
        value[j] = value[j].replace('pulmonary atelectasis', 'atelectasis')
        value[j] = value[j].replace('fractures, bone', 'fracture')
        value[j] = value[j].replace('scolioses', 'scoliosis')
        value[j] = value[j].replace('pleural effusions', 'pleural effusion')
        value[j] = value[j].replace('granulomatous disease', 'granuloma')
        value[j] = value[j].replace('pulmonary emphysema', 'emphysema')
        value[j] = value[j].replace('bilateral pleural effusion', 'pleural effusion')
        value[j] = value[j].replace('aorta, thoracic', 'aorta')
        value[j] = value[j].replace('thoracic aorta', 'aorta').replace('fractures', 'fracture')
        value[j] = value[j].replace('catheterization, central venous', 'central venous catheter')
        value[j] = value[j].replace('central venous catheters', 'central venous catheter')
        value[j] = value[j].replace('pulmonary disease, chronic obstructive', 'copd')
        value[j] = value[j].replace('rib fractures', 'rib fracture')
        value[j] = value[j].replace('chronic lung disease', 'copd')
        value[j] = value[j].replace('right-sided pleural effusion', 'pleural effusion')
        value[j] = value[j].replace('coronary artery bypass', 'cabg')
        value[j] = value[j].replace('focal atelectasis', 'atelectasis')
        value[j] = value[j].replace('chronic granuloma', 'granuloma')
        value[j] = value[j].replace('calcified granuloma', 'granuloma')
        value[j] = value[j].replace('granulomatous infection', 'granuloma')
        value[j] = value[j].replace('spinal osteophytosis', 'osteophytes')
        value[j] = value[j].replace('pulmonary edema', 'edema')
        value[j] = value[j].replace('rib fracture', 'fracture').replace('atheroscleroses', 'atherosclerosis')
        value[j] = value[j].replace('central venous catheter', 'support devices')
        value[j] = value[j].replace('pneumonitis', 'pneumonia')
        value[j] = value[j].replace('right upper lobe pneumonia', 'pneumonia')
        value[j] = value[j].replace('granulomas', 'granuloma').replace('esophagectomies', 'esophagectomy')
        value[j] = value[j].replace('copd, severe early-onset', 'copd')
        value[j] = value[j].replace('hypertension, pulmonary', 'pulmonary hypertension')
        value[j] = value[j].replace('obesity, morbid', 'obesity').replace('catheters', 'catheter').replace('fibroses', 'fibrosis')
        value[j] = value[j].replace('bronchiectases', 'bronchiectasis')
        value[j] = value[j].replace('mitral valve replacement', 'valve replacement').replace('aortic aneurysm, thoracic', 'aortic aneurysm')
        value[j] = value[j].replace('hernia, hiatal', 'hiatal hernia').replace('kyphosis', 'kyphoses')
    value = list(set(value))
    # for tag in value:  # if others is needed
    #     if tag not in tags:
    #         value.append('others')
    # value = list(set(value))

    tag_word = []
    for tag in value:
        if tag in tags:
            tag_word.append(tag)
    diction["test"][key] = tag_word

with open('./mesh_data.json', 'w') as fp:
    json.dump(diction, fp)