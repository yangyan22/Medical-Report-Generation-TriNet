The Pytorch implementaion of the paper: Joint Embedding of Deep Visual and Semantic Features for Medical Image Report Generation, IEEE Transactions on Multimedia, 2021.

Authors: Yan Yang, Jun Yu*, Jian Zhang, Weidong Han*, Hanliang Jiang, and Qingming Huang

If you find our work or our code helpful for your research, please cite our paper.

# Dependencies
  - Python=3.7.3
  - pytorch=1.8.1
  - pickle
  - tqdm
  - time
  - argparse
  - matplotlib
  - sklearn
  - json
  - numpy 
  - torchvision 
  - itertools
  - collections
  - math
  - os
  - matplotlib
  - PIL 
  - itertools
  - copy
  - re
  - abc
  - pandas
  - torch

The ground-truth TF-IDF features of MeSH and MeRP in the training set are constructed before training with codes in TF-IDF folder.

The IF-IDF folder contains:
 1. the build_vocab_TF-IDF.py (for constructing the vocabulary in TF-IDF construction with a vocab_TF-IDF.json)
 2. mesh_tag.py (to select the top 30 MeSH and obtain the MeSH information for each study)
 3. TF_IDF_MeRP.py (to construct the report TF-IDF vector for each study)
 4. TF_IDF_MeSH.py (to construct the MeSh TF-IDF vector for each study)

# reference codes: 
https://github.com/ZexinYan/Medical-Report-Generation

https://github.com/tylin/coco-caption

https://github.com/MorvanZhou/NLP-Tutorials

# the metric meteor
the paraphrase-en.gz should be put into the .\pycocoevalcap\meteor\data, since the file is too big to upload.

