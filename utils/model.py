import torch
import torch.nn as nn
import numpy as np
import torchvision.models as models


class VisualFeatureExtractor(nn.Module):
    def __init__(self, embed_size):
        super(VisualFeatureExtractor, self).__init__()

        # frontal
        resnet_frontal = models.resnet50(pretrained=True)
        self.resnet_conv_frontal = nn.Sequential(*list(resnet_frontal.children())[:-2])
        self.avgpool_fun_frontal = nn.Sequential(* list(resnet_frontal.children())[-2:-1])
        self.dropout_frontal = nn.Dropout(0.2)
        self.affine_frontal_a = nn.Linear(2048, embed_size)
        self.affine_frontal_b = nn.Linear(2048, embed_size)

        # lateral
        resnet_lateral = models.resnet50(pretrained=True)
        self.resnet_conv_lateral = nn.Sequential(*list(resnet_lateral.children())[:-2])
        self.avgpool_fun_lateral = nn.Sequential(* list(resnet_lateral.children())[-2:-1])
        self.dropout_lateral = nn.Dropout(0.2)
        self.affine_lateral_a = nn.Linear(2048, embed_size)
        self.affine_lateral_b = nn.Linear(2048, embed_size)

        self.relu = nn.ReLU()
        self.affine = nn.Linear(2 * embed_size, embed_size)
        self.init_weights()

    def init_weights(self):
        """Initialize the weights."""
        self.affine_frontal_a.weight.data.uniform_(-0.1, 0.1)
        self.affine_frontal_a.bias.data.fill_(0)
        self.affine_frontal_b.weight.data.uniform_(-0.1, 0.1)
        self.affine_frontal_b.bias.data.fill_(0)
        self.affine_lateral_a.weight.data.uniform_(-0.1, 0.1)
        self.affine_lateral_a.bias.data.fill_(0)
        self.affine_lateral_b.weight.data.uniform_(-0.1, 0.1)
        self.affine_lateral_b.bias.data.fill_(0)
        self.affine.weight.data.uniform_(-0.1, 0.1)
        self.affine.bias.data.fill_(0)

    def forward(self, image_frontal, image_lateral):
        A_frontal = self.resnet_conv_frontal(image_frontal)  # [bs, 2048, 7, 7]
        V_frontal = A_frontal.view(A_frontal.size(0), A_frontal.size(1), -1).transpose(1, 2)  # [bs, 49, 2048]
        V_frontal = self.relu(self.affine_frontal_a(self.dropout_frontal(V_frontal)))  # [bs, 49, 512]
        avg_frontal = self.avgpool_fun_frontal(A_frontal).squeeze()  # [bs, 2048]
        avg_frontal = self.relu(self.affine_frontal_b(self.dropout_frontal(avg_frontal)))  # [bs, 512]
        A_lateral = self.resnet_conv_lateral(image_lateral)
        V_lateral = A_lateral.view(A_lateral.size(0), A_lateral.size(1), -1).transpose(1, 2)
        V_lateral = self.relu(self.affine_lateral_a(self.dropout_lateral(V_lateral)))
        avg_lateral = self.avgpool_fun_lateral(A_lateral).squeeze()
        avg_lateral = self.relu(self.affine_lateral_b(self.dropout_lateral(avg_lateral)))
        avg = torch.cat((avg_frontal, avg_lateral), dim=1)
        avg = self.relu(self.affine(avg))
        return V_frontal, V_lateral, avg


class SemanticEmbedding(nn.Module):
    def __init__(self, tag_dim, report_dim, embed_size):
        super(SemanticEmbedding, self).__init__()
        self.mesh_tf = nn.Linear(in_features=embed_size, out_features=tag_dim)
        self.report_tf = nn.Linear(in_features=embed_size, out_features=report_dim)
        self.bn = nn.BatchNorm1d(num_features=embed_size, momentum=0.1)
        self.w1 = nn.Linear(in_features=tag_dim + report_dim, out_features=embed_size)
        self.w2 = nn.Linear(in_features=embed_size, out_features=embed_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.__init_weight()

    def __init_weight(self):
        self.mesh_tf.weight.data.uniform_(-0.1, 0.1)
        self.mesh_tf.bias.data.fill_(0)
        self.report_tf.weight.data.uniform_(-0.1, 0.1)
        self.report_tf.bias.data.fill_(0)
        self.w1.weight.data.uniform_(-0.1, 0.1)
        self.w1.bias.data.fill_(0)
        self.w2.weight.data.uniform_(-0.1, 0.1)
        self.w2.bias.data.fill_(0)

    def forward(self, avg):
        mesh_tf = self.relu(self.mesh_tf(avg))
        report_tf = self.relu(self.report_tf(avg))
        tf = torch.cat((mesh_tf, report_tf), dim=1)
        state_h = self.bn(self.w2(self.w1(tf)))
        return mesh_tf, report_tf, state_h, state_h


class SentenceLSTM(nn.Module):
    def __init__(self,
                 embed_size=512,
                 hidden_size=512):
        super(SentenceLSTM, self).__init__()

        self.lstm = nn.LSTM(input_size=embed_size, hidden_size=hidden_size, batch_first=True)
        self.W_h1 = nn.Linear(in_features=hidden_size, out_features=embed_size, bias=True)
        self.W_h2 = nn.Linear(in_features=hidden_size, out_features=embed_size, bias=True)
        self.W_v1 = nn.Linear(in_features=hidden_size, out_features=embed_size, bias=True)
        self.W_v2 = nn.Linear(in_features=hidden_size, out_features=embed_size, bias=True)
        self.W_1 = nn.Linear(in_features=hidden_size, out_features=1, bias=True)
        self.W_2 = nn.Linear(in_features=hidden_size, out_features=1, bias=True)

        self.W_ctx = nn.Linear(in_features=2 * embed_size, out_features=embed_size, bias=True)
        self.W_output = nn.Linear(in_features=hidden_size, out_features=embed_size, bias=True)
        self.W_stop = nn.Linear(in_features=hidden_size, out_features=2, bias=True)
        self.Wh = nn.Linear(in_features=embed_size, out_features=embed_size, bias=True)
        self.Wc = nn.Linear(in_features=embed_size, out_features=embed_size, bias=True)
        self.dropout = nn.Dropout(0.2)
        self.softmax = nn.Softmax(dim=1)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.__init_weight()

    def __init_weight(self):
        """
        Initializes some parameters with values from the uniform distribution, for easier convergence.
        """
        self.W_h1.weight.data.uniform_(-0.1, 0.1)
        self.W_h1.bias.data.fill_(0)
        self.W_h2.weight.data.uniform_(-0.1, 0.1)
        self.W_h2.bias.data.fill_(0)
        self.W_v1.weight.data.uniform_(-0.1, 0.1)
        self.W_v1.bias.data.fill_(0)
        self.W_v2.weight.data.uniform_(-0.1, 0.1)
        self.W_v2.bias.data.fill_(0)
        self.W_1.weight.data.uniform_(-0.1, 0.1)
        self.W_1.bias.data.fill_(0)
        self.W_2.weight.data.uniform_(-0.1, 0.1)
        self.W_2.bias.data.fill_(0)
        self.W_ctx.weight.data.uniform_(-0.1, 0.1)
        self.W_ctx.bias.data.fill_(0)
        self.W_output.weight.data.uniform_(-0.1, 0.1)
        self.W_output.bias.data.fill_(0)
        self.W_stop.weight.data.uniform_(-0.1, 0.1)
        self.W_stop.bias.data.fill_(0)
        self.Wh.weight.data.uniform_(-0.1, 0.1)
        self.Wh.bias.data.fill_(0)
        self.Wc.weight.data.uniform_(-0.1, 0.1)
        self.Wc.bias.data.fill_(0)

    def forward(self, frontal, lateral, state, phid):

        h1 = self.W_h1(phid)  # [bs, 1, 512]
        h2 = self.W_h2(phid)
        v1 = self.W_v1(frontal)  # [bs, 49, 512]
        v2 = self.W_v2(lateral)

        joint_out1 = self.tanh(torch.add(v1, h1))  # [bs, 49, 512]
        joint_out2 = self.tanh(torch.add(v2, h2))
        join_output1 = self.W_1(joint_out1).squeeze(2)
        join_output2 = self.W_2(joint_out2).squeeze(2)  # [bs, 49]

        alpha_v1 = self.softmax(join_output1)
        alpha_v2 = self.softmax(join_output2)

        ctx1 = torch.sum(frontal * alpha_v1.unsqueeze(2), dim=1)
        ctx2 = torch.sum(lateral * alpha_v2.unsqueeze(2), dim=1)
        ctx = torch.cat((ctx1, ctx2), dim=1)  # [bs, 1024]
        ctx = self.W_ctx(ctx)

        output1, state_t1 = self.lstm(ctx.unsqueeze(1), state)  # [bs, 1, 512] [1, bs, 512]
        p_stop = self.tanh(self.W_stop(output1.squeeze(1)))  # 512->2

        output = self.W_output(output1.squeeze(1))
        topic = self.tanh(torch.add(ctx, output))
        # topic = self.tanh(torch.cat((ctx, output), dim=1))
        h0_word = self.tanh(self.Wh(self.dropout(topic))).unsqueeze(0)
        c0_word = self.tanh(self.Wc(self.dropout(topic))).unsqueeze(0)  # bs 1 512
        phid = output1
        return p_stop, state_t1, h0_word, c0_word, phid


class WordLSTM(nn.Module):
    def __init__(self, embed_size,
                 hidden_size,
                 vocab_size,
                 n_max):
        super(WordLSTM, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(input_size=embed_size, hidden_size=hidden_size, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(p=0.2)
        self.n_max = n_max
        self.vocab_size = vocab_size
        self.__init_weights()

    def __init_weights(self):
        self.embed.weight.data.uniform_(-0.1, 0.1)
        self.linear.weight.data.uniform_(-0.1, 0.1)
        self.linear.bias.data.fill_(0)

    def forward(self, caption, state):
        embeddings = self.embed(caption).unsqueeze(1)
        hidden, states_t = self.lstm(embeddings, state)
        output = self.linear(self.dropout(hidden[:, -1, :]))
        return output, states_t

    def sample(self, start_tokens, state):
        sampled_ids = np.zeros((np.shape(start_tokens)[0], self.n_max))  # BS n_max
        sampled_ids[:, 0] = start_tokens.cpu()
        predicted = start_tokens
        for i in range(1, self.n_max):
            predicted = self.embed(predicted).unsqueeze(1)  # torch.Size([1, 1, 512])
            hidden, state = self.lstm(predicted, state)
            output = self.linear(self.dropout(hidden[:, -1, :]))  
            predicted = torch.max(output, 1)[1]
            sampled_ids[:, i] = predicted.cpu()
        return sampled_ids, predicted

