import time
import argparse
from torch.optim.lr_scheduler import ReduceLROnPlateau
from utils.model import *
from utils.dataset import *
from torch.autograd import Variable
import pickle
import warnings
import torchvision.transforms as transforms
import os
from tqdm import tqdm
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score
from utils.metrics import compute_scores
import pandas as pd


class DebuggerBase:
    def __init__(self, args):
        self.args = args
        self.min_train_loss = 1000
        self.min_train_stop_loss = 1000
        self.min_train_word_loss = 1000
        self.min_train_reg_loss = 1000
        self.max_bleu1 = -0.001
        if not os.path.exists(self.args.model_path):
            os.makedirs(self.args.model_path)
        self.model_dir = self._init_model_dir()
        with open(self.args.vocab_path, 'rb') as f:
            self.vocab = pickle.load(f)
        self.train_transform = self._init_train_transform()
        self.val_transform = self._init_val_transform()
        self.test_transform = self._init_test_transform()
        self.train_data_loader = self._init_data_loader(split='train', transform=self.train_transform, shuffle=True)
        self.val_data_loader = self._init_data_loader(split='val', transform=self.val_transform, shuffle=False)
        self.test_data_loader = self._init_data_loader(split='test', transform=self.test_transform, shuffle=False)
        self.ce_criterion = self._init_ce_criterion()
        self.l1_criterion = self._init_l1_criterion()
        self.mse_criterion = self._init_mse_criterion()
        self.model_state_dict = self._load_mode_state_dict()

        self.params = None
        self.extractor = self._init_visual_extractor()
        self.semantic = self._init_semantic_embedding()
        self.sentence_model = self._init_sentence_model()
        self.word_model = self._init_word_model()
        self.optimizer = torch.optim.Adam(params=self.params, lr=self.args.learning_rate, weight_decay=0, amsgrad=False)
        self.scheduler = ReduceLROnPlateau(self.optimizer, 'min', patience=self.args.patience, factor=0.1)
        self.epochs_recorder = {}

    def _print_epochs_to_file(self):
        self.epochs_recorder['time'] = time.asctime(time.localtime(time.time()))
        record_path = os.path.join(self.model_dir, 'print_epochs.csv')
        print("record_path : {}".format(record_path))
        if not os.path.exists(record_path):
            record_table = pd.DataFrame()
        else:
            record_table = pd.read_csv(record_path)
        self.epochs_recorder["test_BO_M"] = self.epochs_recorder["test_METEOR"]
        self.epochs_recorder["test_BP_R"] = self.epochs_recorder["test_ROUGE_L"]
        self.epochs_recorder["test_BQ_C"] = self.epochs_recorder["test_CIDEr"]

        self.epochs_recorder["val_BO_M"] = self.epochs_recorder["val_METEOR"]
        self.epochs_recorder["val_BP_R"] = self.epochs_recorder["val_ROUGE_L"]
        self.epochs_recorder["val_BQ_C"] = self.epochs_recorder["val_CIDEr"]
        record_table = record_table.append(self.epochs_recorder, ignore_index=True)
        record_table.to_csv(record_path, index=False)

    def _init_model_dir(self):
        model_dir = self.args.model_path
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        model_dir = os.path.join(model_dir, self._get_now())
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        return model_dir

    def _init_train_transform(self):
        transform = transforms.Compose([
            transforms.Resize(self.args.resize),
            transforms.RandomCrop(self.args.size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406),
                                 (0.229, 0.224, 0.225))])
        return transform

    def _init_val_transform(self):
        transform = transforms.Compose([
            transforms.Resize((self.args.size, self.args.size)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406),
                                 (0.229, 0.224, 0.225))])
        return transform

    def _init_test_transform(self):
        transform = transforms.Compose([
            transforms.Resize((self.args.size, self.args.size)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406),
                                 (0.229, 0.224, 0.225))])
        return transform

    def _load_mode_state_dict(self):
        self.start_epoch = 0
        try:
            model_state = torch.load(self.args.load_model_path)
            self.start_epoch = model_state['epoch'] + 1
            print("[Load Model {} Succeed!]\n".format(self.args.load_model_path))
            print("Load From Epoch {}\n".format(model_state['epoch']))
            return model_state
        except Exception as err:
            print("[Load Model Failed] {}\n".format(err))
            return None

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

    @staticmethod
    def _init_ce_criterion():
        return nn.CrossEntropyLoss(reduction='mean')

    @staticmethod
    def _init_l1_criterion():
        return nn.L1Loss(size_average=True, reduce=True)

    @staticmethod
    def _init_mse_criterion():
        return nn.MSELoss(reduction='sum')

    def _to_var(self, x, requires_grad=True):
        if self.args.cuda:
            x = x.cuda()
        return Variable(x, requires_grad=requires_grad)

    def _get_date(self):
        return str(time.strftime('%Y-%m-%d', time.localtime()))

    def _get_now(self):
        return str(time.strftime('%Y-%m-%d %H:%M', time.localtime()))

    def _save_model(self, epoch_id, train_loss, b1, b2, b3, b4, ROUGE_L, CIDEr, METEOR):
        def save_whole_model(_filename):
            print("Saved Model in {}".format(_filename))
            torch.save({'extractor': self.extractor.state_dict(),
                        'semantic': self.semantic.state_dict(),
                        'sentence_model': self.sentence_model.state_dict(),
                        'word_model': self.word_model.state_dict(),
                        'optimizer': self.optimizer.state_dict(),
                        'epoch': epoch_id},
                         os.path.join(self.model_dir, "{}".format(_filename)))

        if train_loss < self.min_train_loss:
            file_name = "train_best.pth.tar"
            save_whole_model(file_name)
            self.min_train_loss = train_loss

        if b1 > self.max_bleu1:
            file_name = "val_best.pth.tar"
            save_whole_model(file_name)
            self.max_bleu1 = b1
            self.max_bleu2 = b2
            self.max_bleu3 = b3
            self.max_bleu4 = b4
            self.max_METEOR = METEOR
            self.max_ROUGE_L = ROUGE_L
            self.max_CIDEr = CIDEr

        if epoch_id % self.args.save_period == 0 and epoch_id != 0:
            file_name = "save{}.pth.tar".format(epoch_id)
            save_whole_model(file_name)

    def _init_visual_extractor(self):
        model = VisualFeatureExtractor(self.args.embed_size)
        try:
            model_state = torch.load(self.args.load_visual_model_path)
            model.load_state_dict(model_state['extractor'])
            print("[Load Visual Extractor Succeed!]\n")
        except Exception as err:
            print("[Load Model Failed] {}\n".format(err))

        if not self.args.visual_trained:
            for i, param in enumerate(model.parameters()):
                param.requires_grad = False
        else:
            if self.params:
                self.params += list(model.parameters())
            else:
                self.params = list(model.parameters())
        if self.args.cuda:
            model = model.cuda()
        return model

    def _init_semantic_embedding(self):
        model = SemanticEmbedding(mesh_dim=self.args.mesh_dim,
                                  report_dim=self.args.report_dim,
                                  embed_size=self.args.embed_size)
        try:
            model_state = torch.load(self.args.load_semantic_model_path)
            model.load_state_dict(model_state['semantic'])
            print("[Load Semantic Embedding Succeed!]\n")
        except Exception as err:
            print("[Load Semantic Embedding Failed {}!]\n".format(err))
        if not self.args.semantic_trained:
            for i, param in enumerate(model.parameters()):
                param.requires_grad = False
        else:
            if self.params:
                self.params += list(model.parameters())
            else:
                self.params = list(model.parameters())
        if self.args.cuda:
            model = model.cuda()
        return model

    def _init_sentence_model(self):
        model = SentenceLSTM(embed_size=self.args.embed_size,
                             hidden_size=self.args.hidden_size)
        try:
            model_state = torch.load(self.args.load_sentence_model_path)
            model.load_state_dict(model_state['sentence_model'])
            print("[Load Sentence Model Succeed!]\n")
        except Exception as err:
            print("[Load Sentence model Failed {}!]\n".format(err))
        if not self.args.sentence_trained:
            for i, param in enumerate(model.parameters()):
                param.requires_grad = False
        else:
            if self.params:
                self.params += list(model.parameters())
            else:
                self.params = list(model.parameters())

        if self.args.cuda:
            model = model.cuda()
        return model

    def _init_word_model(self):
        model = WordLSTM(embed_size=self.args.embed_size,
                         hidden_size=self.args.hidden_size,
                         vocab_size=len(self.vocab),
                         n_max=self.args.n_max)
        try:
            model_state = torch.load(self.args.load_word_model_path)
            model.load_state_dict(model_state['word_model'])
            print("[Load Word Model Succeed!]\n")
        except Exception as err:
            print("[Load Word model Failed {}!]\n".format(err))

        if not self.args.word_trained:
            for i, param in enumerate(model.parameters()):
                param.requires_grad = False
        else:
            if self.params:
                self.params += list(model.parameters())
            else:
                self.params = list(model.parameters())
        if self.args.cuda:
            model = model.cuda()
        return model

    def train(self):
        loss_train = []
        loss_word = []
        loss_semantic = []
        loss_stop = []
        BLEU1 = []
        BLEU2 = []

        for epoch_id in range(self.start_epoch, self.args.epochs):
            print('Epoch:{}'.format(epoch_id))
            log = {'epoch': epoch_id}
            stop_loss, word_loss, semantic_loss, train_loss = self._epoch_train()
            val_met, val_acc = self._epoch_val()
            test_met, test_acc = self._epoch_test()
            b1 = test_met['BLEU_1']
            b2 = test_met['BLEU_2']
            b3 = test_met['BLEU_3']
            b4 = test_met['BLEU_4']
            CIDEr = test_met['CIDEr']
            ROUGE_L = test_met['ROUGE_L']
            METEOR = test_met['METEOR']

            loss_stop.append(stop_loss)
            loss_word.append(word_loss)
            loss_semantic.append(semantic_loss)
            loss_train.append(train_loss)

            plt.plot(loss_semantic, color='red', label='loss_semantic')
            plt.plot(loss_train, color='blue', label='loss_train')
            plt.plot(loss_word, color='black', label='loss_word')
            plt.plot(loss_stop, color='purple', label='loss_stop')
            plt.title('loss_plot')
            plt.savefig(self.model_dir + "/loss_plot_train.png")
            plt.close('all')

            BLEU1.append(b1)
            BLEU2.append(b2)
            plt.plot(BLEU1, color='red', label='BLUE1')
            plt.plot(BLEU2, color='black', label='BLUE2')
            plt.savefig(self.model_dir + "/BLEU.png")
            plt.close('all')

            self._save_model(epoch_id, train_loss, b1, b2, b3, b4, ROUGE_L, CIDEr, METEOR)
            print('BLEU1:{}'.format(self.max_bleu1))
            print('BLEU2:{}'.format(self.max_bleu2))
            print('BLEU3:{}'.format(self.max_bleu3))
            print('BLEU4:{}'.format(self.max_bleu4))
            print('METEOR:{}'.format(self.max_METEOR))
            print('ROUGE_L:{}'.format(self.max_ROUGE_L))
            print('CIDEr:{}'.format(self.max_CIDEr))
            print("\n")
            self.scheduler.step(train_loss)
            log.update(**{'test_' + k: v for k, v in test_met.items()})
            log.update(**{'val_' + k: v for k, v in val_met.items()})
            log.update(**{'test_' + k: v for k, v in test_met.items()})
            # log.update(**test_acc)
            # log.update(**val_acc)
            self.epochs_recorder.update(log)
            self._print_epochs_to_file()


class LSTMDebugger(DebuggerBase):
    def _init_(self, args):
        DebuggerBase.__init__(self, args)
        self.args = args

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

    def _epoch_train(self):
        self.extractor.train()
        self.semantic.train()
        self.sentence_model.train()
        self.word_model.train()
        sto = []
        wor = []
        sem = []
        epoc = []
        start_time = time.time()
        progress_bar = tqdm(self.train_data_loader, desc='Training')
        for i, (images1, images2, targets, prob, mesh, report, id) in enumerate(progress_bar):
            captions = self._to_var(torch.Tensor(targets).long(), requires_grad=False)
            prob_real = self._to_var(torch.Tensor(prob).long(), requires_grad=False)
            gt_tf_mesh = self._to_var(torch.Tensor(mesh), requires_grad=False)
            gt_tf_report = self._to_var(torch.Tensor(report), requires_grad=False)
            images_frontal = self._to_var(images1, requires_grad=False)
            images_lateral = self._to_var(images2, requires_grad=False)
            batch_stop_loss, batch_word_loss, batch_report_loss, batch_mesh_loss, batch_loss = 0, 0, 0, 0, 0

            frontal, lateral, avg = self.extractor.forward(images_frontal, images_lateral)
            mesh_tf, report_tf, state_c, state_h = self.semantic.forward(avg)
            state = (torch.unsqueeze(state_c, 0), torch.unsqueeze(state_h, 0))
            pre_hid = torch.unsqueeze(state_h, 1)
            for sentence_index in range(captions.shape[1]):
                p_stop, state, h0_word, c0_word, pre_hid = self.sentence_model.forward(frontal, lateral, state, pre_hid)
                batch_stop_loss += self.ce_criterion(p_stop.squeeze(), prob_real[:, sentence_index]).sum()
                state_word = (c0_word, h0_word)
                for word_index in range(captions.shape[2]-1):
                    word, state_word = self.word_model.forward(captions[:, sentence_index, word_index], state_word)
                    word_mask = (captions[:, sentence_index, word_index + 1] > 0).float()
                    batch_word_loss += (self.ce_criterion(word, captions[:, sentence_index, word_index + 1]) * word_mask).mean()

            loss_tf_mesh = self.mse_criterion(nn.functional.normalize(gt_tf_mesh), nn.functional.normalize(mesh_tf))
            loss_tf_report = self.mse_criterion(nn.functional.normalize(gt_tf_report), nn.functional.normalize(report_tf))
            batch_stop_loss = batch_stop_loss * 0.5
            batch_word_loss = batch_word_loss
            loss_semantic = (loss_tf_report + loss_tf_mesh)
            batch_loss = (batch_word_loss + loss_semantic + batch_stop_loss) / 2
            sto.append(batch_stop_loss.item())
            wor.append(batch_word_loss.item())
            sem.append(loss_semantic.item())
            epoc.append(batch_loss.item())

            self.optimizer.zero_grad()
            batch_loss.backward()

            if self.args.clip > 0:
                torch.nn.utils.clip_grad_norm(self.sentence_model.parameters(), self.args.clip)
                torch.nn.utils.clip_grad_norm(self.word_model.parameters(), self.args.clip)
            self.optimizer.step()
        end_time = time.time()
        stop_loss = np.mean(sto)
        word_loss = np.mean(wor)
        semantic_loss = np.mean(sem)
        train_loss = np.mean(epoc)
        print("time:%d" % (end_time - start_time))
        print('train_stop_loss:{}'.format(stop_loss))
        print('train_word_loss:{}'.format(word_loss))
        print('batch_report_loss:{}'.format(semantic_loss))
        print('train_epoch_loss:{}'.format(train_loss))
        return stop_loss, word_loss, semantic_loss, train_loss

    def _epoch_val(self):
        self.extractor.eval()
        self.sentence_model.eval()
        self.word_model.eval()
        self.semantic.eval()
        start_time = time.time()
        progress_bar = tqdm(self.val_data_loader, desc='Generating')
        results = {}
        acc = []
        mesh_vocab = ['normal', 'degenerative change', 'granuloma', 'opacity', 'atelectasis', 'cardiomegaly', 'scar', 'pleural effusion', 'aorta', 'fracture', 'emphysema', 'pneumonia', 'sternotomy', 'diaphragm', 'nodule', 'infiltrates', 'deformity', 'osteophytes', 'copd', 'edema', 'support devices', 'eventration', 'thoracic vertebrae', 'tortuous aorta', 'cabg', 'scoliosis', 'hyperinflation', 'calcinosis', 'hiatal hernia', 'effusion']
        for images1, images2, targets, prob, mesh, report, study in progress_bar:
            images_frontal = self._to_var(images1, requires_grad=False)
            images_lateral = self._to_var(images2, requires_grad=False)
            frontal, lateral, avg = self.extractor.forward(images_frontal, images_lateral)  # [8, 49, 512] [8, 512]
            mesh_tf, report_tf, state_c, state_h = self.semantic.forward(avg)  # [BS, 30]
            gt_mesh = self._to_var(torch.Tensor(mesh), requires_grad=False).cpu().numpy()
            true = np.array(gt_mesh > 0.0, dtype=float)
            pred = np.array(mesh_tf.detach().cpu().numpy() > 0.5, dtype=float)
            res = precision_score(y_true=true, y_pred=pred, average='micro')
            acc.append(res)
            pred_sentences = {}
            real_sentences = {}
            pred_tag = {}
            real_tag = {}
            for i in study:
                pred_sentences[i] = {}
                real_sentences[i] = {}
                pred_tag[i] = {}
                real_tag[i] = {}
            state = (torch.unsqueeze(state_c, 0), torch.unsqueeze(state_h, 0))
            phid = torch.unsqueeze(state_h, 1)
            for sentence_index in range(self.args.s_max):
                p_stop, state, h0_word, c0_word, phid = self.sentence_model.forward(frontal, lateral, state, phid)
                p_stop = p_stop.squeeze(1)
                p_stop = torch.unsqueeze(torch.max(p_stop, 1)[1], 1)
                states_word = (c0_word, h0_word)
                start_tokens = np.zeros(images_frontal.shape[0])
                start_tokens[:] = self.vocab('<start>')
                start_tokens = self._to_var(torch.Tensor(start_tokens).long(), requires_grad=False)
                sampled_ids, _ = self.word_model.sample(start_tokens, states_word)
                sampled_ids = sampled_ids * p_stop.cpu().numpy()
                for id, array in zip(study, sampled_ids):
                    pred_sentences[id][sentence_index] = self.__vec2sent(array)

            for id, array in zip(study, targets):
                for i, sent in enumerate(array):
                    real_sentences[id][i] = self.__vec2sent(sent)

            for id, array in zip(study, pred):
                pred_tag[id] = array

            for id, array in zip(study, mesh):
                real_tag[id] = array

            for id in study:
                # print(id)
                # print('Pred Sent.{}'.format(pred_sentences[id]))
                # print('Real Sent.{}'.format(real_sentences[id]))
                # print('\n')
                results[id] = {'Pred Sent': pred_sentences[id], 'Real Sent': real_sentences[id]}

            pred_tags = []
            for i in range(self.args.mesh_dim):
                if pred_tag[id][i] == 1:
                    pred_tags.append(mesh_vocab[i])

            real_tags = []
            for i in range(self.args.mesh_dim):
                if real_tag[id][i] == 1:
                    real_tags.append(mesh_vocab[i])
            # print("\n")
            # print('Pred Tags.{}'.format(pred_tags))
            # print('Real Tags.{}'.format(real_tags))
            # print('Pred Sent.{}'.format(pred_sentences[id]))
            # print('Real Sent.{}'.format(real_sentences[id]))
                # result_path = os.path.join(self.args.model_dir, self.args.result_path)
                # if not os.path.exists(result_path):
                #     os.makedirs(result_path)
                # with open(os.path.join(result_path, '{}.json'.format(self.args.result_name)), 'w') as f:
                #     json.dump(results, f)
        end_time = time.time()
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

        val_met = compute_scores({i: [gt] for i, gt in enumerate(gts)},
                                  {i: [re] for i, re in enumerate(res)})
        print(val_met)
        accuracy = np.mean(acc)
        print('VAL_Acc:{}'.format(accuracy))
        print("time:%d" % (end_time - start_time))
        return val_met, accuracy

    def _epoch_test(self):
        self.extractor.eval()
        self.sentence_model.eval()
        self.word_model.eval()
        self.semantic.eval()
        start_time = time.time()
        progress_bar = tqdm(self.test_data_loader, desc='Generating')
        results = {}
        acc = []
        mesh_vocab = ['normal', 'degenerative change', 'granuloma', 'opacity', 'atelectasis', 'cardiomegaly', 'scar', 'pleural effusion', 'aorta', 'fracture', 'emphysema', 'pneumonia', 'sternotomy', 'diaphragm', 'nodule', 'infiltrates', 'deformity', 'osteophytes', 'copd', 'edema', 'support devices', 'eventration', 'thoracic vertebrae', 'tortuous aorta', 'cabg', 'scoliosis', 'hyperinflation', 'calcinosis', 'hiatal hernia', 'effusion']
        for images1, images2, targets, prob, mesh, report, study in progress_bar:
            images_frontal = self._to_var(images1, requires_grad=False)
            images_lateral = self._to_var(images2, requires_grad=False)
            frontal, lateral, avg = self.extractor.forward(images_frontal, images_lateral)  # [8, 49, 512] [8, 512]
            mesh_tf, report_tf, state_c, state_h = self.semantic.forward(avg)  # [BS, 30]
            gt_mesh = self._to_var(torch.Tensor(mesh), requires_grad=False).cpu().numpy()
            true = np.array(gt_mesh > 0.0, dtype=float)
            pred = np.array(mesh_tf.detach().cpu().numpy() > 0.5, dtype=float)
            res = precision_score(y_true=true, y_pred=pred, average='micro')
            acc.append(res)
            pred_sentences = {}
            real_sentences = {}
            pred_tag = {}
            real_tag = {}
            for i in study:
                pred_sentences[i] = {}
                real_sentences[i] = {}
                pred_tag[i] = {}
                real_tag[i] = {}
            state = (torch.unsqueeze(state_c, 0), torch.unsqueeze(state_h, 0))
            phid = torch.unsqueeze(state_h, 1)
            for sentence_index in range(self.args.s_max):
                p_stop, state, h0_word, c0_word, phid = self.sentence_model.forward(frontal, lateral, state, phid)
                p_stop = p_stop.squeeze(1)
                p_stop = torch.unsqueeze(torch.max(p_stop, 1)[1], 1)
                states_word = (c0_word, h0_word)
                start_tokens = np.zeros(images_frontal.shape[0])
                start_tokens[:] = self.vocab('<start>')
                start_tokens = self._to_var(torch.Tensor(start_tokens).long(), requires_grad=False)
                sampled_ids, _ = self.word_model.sample(start_tokens, states_word)
                sampled_ids = sampled_ids * p_stop.cpu().numpy()
                for id, array in zip(study, sampled_ids):
                    pred_sentences[id][sentence_index] = self.__vec2sent(array)

            for id, array in zip(study, targets):
                for i, sent in enumerate(array):
                    real_sentences[id][i] = self.__vec2sent(sent)

            for id, array in zip(study, pred):
                pred_tag[id] = array

            for id, array in zip(study, mesh):
                real_tag[id] = array

            for id in study:
                print(id)
                print('Pred Sent.{}'.format(pred_sentences[id]))
                print('Real Sent.{}'.format(real_sentences[id]))
                print('\n')
                results[id] = {'Pred Sent': pred_sentences[id], 'Real Sent': real_sentences[id]}

            pred_tags = []
            for i in range(self.args.mesh_dim):
                if pred_tag[id][i] == 1:
                    pred_tags.append(mesh_vocab[i])

            real_tags = []
            for i in range(self.args.mesh_dim):
                if real_tag[id][i] == 1:
                    real_tags.append(mesh_vocab[i])
            # print("\n")
            # print('Pred Tags.{}'.format(pred_tags))
            # print('Real Tags.{}'.format(real_tags))
            # print('Pred Sent.{}'.format(pred_sentences[id]))
            # print('Real Sent.{}'.format(real_sentences[id]))
            # result_path = os.path.join(self.args.model_dir, self.args.result_path)
            # if not os.path.exists(result_path):
            #     os.makedirs(result_path)
            # with open(os.path.join(result_path, '{}.json'.format(self.args.result_name)), 'w') as f:
            #     json.dump(results, f)
        end_time = time.time()
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
        print("Time:%d" % (end_time - start_time))
        return test_met, accuracy


if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    parser = argparse.ArgumentParser()
    parser.add_argument('--patience', type=int, default=10)  # patience for updating the lr
    DATA_Path = '/media/camlab1/doc_drive/IU_data/images_R2_Ori'

    parser.add_argument('--vocab_path', type=str, default=DATA_Path + '/vocab.pkl', help='path for vocabulary')
    parser.add_argument('--data_dir', type=str, default=DATA_Path + '/iu_annotation_R2Gen.json', help='path for images')

    parser.add_argument('--mesh_tf_idf', type=str, default=DATA_Path + '/TF_IDF_Mesh.json', help='mesh_tf_idf')
    parser.add_argument('--report_tf_idf', type=str, default=DATA_Path + '/TF_IDF_Report.json', help='report_tf_idf')
    parser.add_argument('--model_path', type=str, default='./models', help='path for saving models')

    RESUME_MODEL_PATH = './models/train_best.pth.tar'
    parser.add_argument('--load_model_path', type=str, default=RESUME_MODEL_PATH, help='path of resume')
    parser.add_argument('--save_period', type=int, default=10, help='period of saving the model')

    # Transforms
    parser.add_argument('--resize', type=int, default=256, help='size for resizing images')
    parser.add_argument('--size', type=int, default=224, help='size for randomly cropping images')

    # VisualFeatureExtractor
    parser.add_argument('--load_visual_model_path', type=str, default=RESUME_MODEL_PATH)
    parser.add_argument('--visual_trained', action='store_true', default=True, help='visual extractor or not')

    # SemanticFeatureEmbedding
    parser.add_argument('--mesh_dim', type=int, default=30)  # MeSH embedding dim: the length of MeSH TF-IDF
    parser.add_argument('--report_dim', type=int, default=800)  # MeRP embedding dim: the length of report TF-IDF 
    parser.add_argument('--fc_in_features', type=int, default=1024)
    parser.add_argument('--load_semantic_model_path', type=str, default=RESUME_MODEL_PATH)
    parser.add_argument('--semantic_trained', action='store_true', default=True)

    # Sentence Model
    parser.add_argument('--embed_size', type=int, default=512)
    parser.add_argument('--hidden_size', type=int, default=512)
    parser.add_argument('--load_sentence_model_path', type=str, default=RESUME_MODEL_PATH)
    parser.add_argument('--sentence_trained', action='store_true', default=True)

    # Word Model
    parser.add_argument('--s_max', type=int, default=7)
    parser.add_argument('--n_max', type=int, default=20)
    parser.add_argument('--load_word_model_path', type=str, default=RESUME_MODEL_PATH)
    parser.add_argument('--word_trained', action='store_true', default=True)

    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--learning_rate', type=int, default=5e-5)
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--clip', type=float, default=1, help='gradient clip, -1 means no clip (default: 0.35)')
    parser.add_argument('--result_path', type=str, default='results', help='the path for storing results')
    parser.add_argument('--result_name', type=str, default='results', help='the name of json results')

    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()
    debugger = LSTMDebugger(args)
    debugger.train()
