import argparse
from tqdm import tqdm
from utils.model import *
from utils.dataset import *
from sklearn.metrics import precision_score
from utils.metrics import compute_scores
from torch.autograd import Variable
import os


class CaptionSampler(object):
    def __init__(self, args):
        self.args = args
        self.vocab = self.__init_vocab()
        self.test_transform = self.__init_transform()
        self.test_data_loader = self._init_data_loader(split='test', transform=self.test_transform, shuffle=False)
        self.load_model_path = os.path.join(self.args.model_dir, self.args.load_model_path)
        self.model_state_dict = self.__load_mode_state_dict()
        self.extractor = self.__init_visual_extractor()
        self.semantic = self._init_semantic_embedding()
        self.sentence_model = self.__init_sentence_model()
        self.word_model = self.__init_word_word()

    def __init_vocab(self):
        with open(self.args.vocab_path, 'rb') as f:
            vocab = pickle.load(f)
        # print("vocab_size: ", len(vocab))
        return vocab

    def __init_transform(self):
        transform = transforms.Compose([
            transforms.Resize((self.args.resize, self.args.resize)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406),
                                 (0.229, 0.224, 0.225))])
        return transform

    def _init_data_loader(self, split, transform, shuffle):
        data_loader = get_loader(data_dir=self.args.data_dir,
                                 split=split,
                                 vocabulary=self.vocab,
                                 MeSH_path=self.args.mesh_tf_idf,
                                 Report_path=self.args.report_tf_idf,
                                 transform=transform,
                                 batch_size=self.args.batch_size,
                                 s_max=self.args.s_max,
                                 n_max=self.args.n_max,
                                 shuffle=shuffle)
        return data_loader

    def __load_mode_state_dict(self):
        try:
            model_state_dict = torch.load(self.load_model_path)
            print("[Load Model {} Succeed!]  ".format(self.load_model_path))
            print("Load From Epoch {}".format(model_state_dict['epoch']))
            return model_state_dict
        except Exception as err:
            print("[Load Model Failed] {}".format(err))
            raise err

    def __init_visual_extractor(self):
        model = VisualFeatureExtractor(self.args.embed_size)
        if self.model_state_dict is not None:
            model.load_state_dict(self.model_state_dict['extractor'])
            print("Visual Extractor Loaded!")
        if self.args.cuda:
            model = model.cuda()
        return model

    def _init_semantic_embedding(self):
        model = SemanticEmbedding(tag_dim=self.args.tag_dim, report_dim=self.args.report_dim, embed_size=self.args.embed_size)
        if self.model_state_dict is not None:
            model.load_state_dict(self.model_state_dict['semantic'])
            print("Semantic Embedding Loaded!")
        if self.args.cuda:
            model = model.cuda()
        return model

    def __init_sentence_model(self):
        model = SentenceLSTM(embed_size=self.args.embed_size,
                             hidden_size=self.args.hidden_size)
        if self.model_state_dict is not None:
            model.load_state_dict(self.model_state_dict['sentence_model'])
            print("Sentence Model Loaded!")
        if self.args.cuda:
            model = model.cuda()
        return model

    def __init_word_word(self):
        model = WordLSTM(embed_size=self.args.embed_size,
                         hidden_size=self.args.hidden_size,
                         vocab_size=len(self.vocab),
                         n_max=self.args.n_max)
        if self.model_state_dict is not None:
            model.load_state_dict(self.model_state_dict['word_model'])
            print("Word Model Loaded!")
        if self.args.cuda:
            model = model.cuda()
        return model

    def _to_var(self, x, requires_grad=False):
        if self.args.cuda:
            x = x.cuda()
        return Variable(x, requires_grad=requires_grad)

    def __vec2sent(self, array):  
        sampled_caption = []
        for word_id in array:
            word = self.vocab.get_word_by_id(word_id)
            if word == '<start>':
                continue
            if word == '<end>' or word == '<pad>':
                break
            sampled_caption.append(word)
        return ' '.join(sampled_caption)

    def generate(self):
        self.extractor.eval()
        self.sentence_model.eval()
        self.word_model.eval()
        self.semantic.eval()
        progress_bar = tqdm(self.test_data_loader, desc='Generating')
        results = {}
        acc = []
        tag_vocab = ['normal', 'degenerative change', 'opacity', 'granuloma', 'atelectasis', 'cardiomegaly', 'scar', 'pleural effusion', 'aorta', 'fracture', 'sternotomy', 'emphysema', 'pneumonia', 'infiltrates', 'osteophytes', 'copd', 'nodule', 'edema', 'deformity', 'diaphragm', 'thoracic vertebrae', 'cabg', 'arthritic changes', 'hiatal hernia', 'support devices', 'hyperinflation', 'tortuous aorta', 'congestion', 'hyperexpansion', 'scoliosis', 'others']

        for images1, images2, targets, prob, mesh, report, study_id in progress_bar:
            images_frontal = self._to_var(images1)
            images_lateral = self._to_var(images2)
            frontal, lateral, avg = self.extractor.forward(images_frontal, images_lateral)  # [8, 49, 512] [8, 512]
            mesh_tf, report_tf, state_c, state_h = self.semantic.forward(avg)  # [BS, 30]

            gt_tags = self._to_var(torch.Tensor(mesh))
            true = gt_tags.cpu().numpy()
            true = np.array(true > 0.0, dtype=float)
            pred = mesh_tf.detach().cpu().numpy()
            pred = np.array(pred > 0.8, dtype=float)
            res = precision_score(y_true=true, y_pred=pred, average='micro')
            acc.append(res)

            pred_sentences = {}
            real_sentences = {}
            pred_tag = {}
            real_tag = {}
            for i in study_id:
                pred_sentences[i] = {}
                real_sentences[i] = {}
                pred_tag[i] = {}
                real_tag[i] = {}
            state = (state_c.unsqueeze(0), state_h.unsqueeze(0))
            phid = state_h.unsqueeze(1)  # torch.Size([16, 1, 512])
            for sentence_index in range(self.args.s_max):
                p_stop, state, h0_word, c0_word, phid = self.sentence_model.forward(frontal, lateral, state, phid)
                p_stop = p_stop.squeeze(1)  # BS 1 2->BS 2
                p_stop = torch.max(p_stop, 1)[1].unsqueeze(1)  # 0/1 

                states_word = (h0_word, c0_word)
                start_tokens = np.zeros(images_frontal.shape[0])
                start_tokens[:] = self.vocab('<start>')  # [16]
                start_tokens = self._to_var(torch.Tensor(start_tokens).long())

                sampled_ids, _ = self.word_model.sample(start_tokens, states_word)  
                sampled_ids = sampled_ids * p_stop.cpu().numpy()

                for id, array in zip(study_id, sampled_ids):
                    pred_sentences[id][sentence_index] = self.__vec2sent(array)

            for id, array in zip(study_id, targets):
                for i, sent in enumerate(array):
                    real_sentences[id][i] = self.__vec2sent(sent)

            for id, array in zip(study_id, pred):  
                pred_tag[id] = array

            for id, array in zip(study_id, mesh):
                real_tag[id] = array

            for id in study_id:
                # print('Pred Sent.{}'.format(pred_sentences[id]))
                # print('Real Sent.{}'.format(real_sentences[id]))
                # print('\n')
                results[id] = {'Pred Sent': pred_sentences[id], 'Real Sent': real_sentences[id]}

            pred_tags = []
            for i in range(31):
                if pred_tag[id][i] == 1:
                    pred_tags.append(tag_vocab[i])

            real_tags = []
            for i in range(31):
                if real_tag[id][i] == 1:
                    real_tags.append(tag_vocab[i])

            print("\n")
            print(id)
            print('Pred Tags.{}'.format(pred_tags))
            print('Real Tags.{}'.format(real_tags))
            print('Pred Sent.{}'.format(pred_sentences[id]))
            print('Real Sent.{}'.format(real_sentences[id]))

        result_path = os.path.join(self.args.model_dir, self.args.result_path)
        if not os.path.exists(result_path):
            os.makedirs(result_path)
        with open(os.path.join(result_path, '{}.json'.format(self.args.result_name)), 'w') as f:
            json.dump(results, f)

        gts = []
        res = []
        for key in results:
            gt = ""
            re = ""
            for i in results[key]["Real Sent"]:
                if results[key]["Real Sent"][i] != "":
                    gt = gt + results[key]["Real Sent"][i] + " . "

            for i in results[key]["Pred Sent"]:
                if results[key]["Pred Sent"][i] != "":
                    re = re + results[key]["Pred Sent"][i] + " . "
            gts.append(gt)
            res.append(re)

        test_met = compute_scores({i: [gt] for i, gt in enumerate(gts)},
                                  {i: [re] for i, re in enumerate(res)})
        print(test_met)
        accuracy = np.mean(acc)
        print('TEST_Acc:{}'.format(accuracy))


if __name__ == '__main__':
    import warnings
    warnings.filterwarnings("ignore")
    parser = argparse.ArgumentParser()
    DATA_Path = "/media/camlab1/doc_drive/IU_data/YY/YY_Data"
    parser.add_argument('--vocab_path', type=str, default=DATA_Path+"/vocab.pkl", help='path of vocabulary')
    parser.add_argument('--data_dir', type=str, default=DATA_Path+"/data.json", help='path of data')
    parser.add_argument('--mesh_tf_idf', type=str, default=DATA_Path + "/TF_IDF_Mesh.json", help='path of mesh_tf_idf')
    parser.add_argument('--report_tf_idf', type=str, default=DATA_Path + "/TF_IDF_Report.json", help='path of report_tf_idf')

    parser.add_argument('--model_dir', type=str, default='./models/2021-06-15 21:57/', help='path of model')
    parser.add_argument('--load_model_path', type=str, default='val_best_loss.pth.tar', help='path of trained model')
    parser.add_argument('--resize', type=int, default=224, help='size for resizing images')
    parser.add_argument('--result_path', type=str, default='results', help='the path for storing results')
    parser.add_argument('--result_name', type=str, default='results', help='the name of json results')

    parser.add_argument('--tag_dim', type=int, default=30)
    parser.add_argument('--report_dim', type=int, default=800)
    parser.add_argument('--embed_size', type=int, default=512)
    parser.add_argument('--hidden_size', type=int, default=512)
    parser.add_argument('--s_max', type=int, default=7)
    parser.add_argument('--n_max', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=16)

    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()
    sampler = CaptionSampler(args)
    sampler.generate()
