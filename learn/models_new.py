
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_ as xavier_uniform
from elmo.elmo import Elmo
from utils import build_pretrain_embedding, load_embeddings
import json
from learn.transformer import EncoderHidden
import numpy as np
from learn.focal_loss import FocalLoss
from math import floor

class WordRep(nn.Module):
    def __init__(self, args, Y, dicts):
        super(WordRep, self).__init__()

        self.gpu = args.gpu

        if args.embed_file:
            print("loading pretrained embeddings from {}".format(args.embed_file))
            if args.use_ext_emb:
                pretrain_word_embedding, pretrain_emb_dim = build_pretrain_embedding(args.embed_file, dicts['w2ind'],
                                                                                     True)
                W = torch.from_numpy(pretrain_word_embedding)
            else:
                W = torch.Tensor(load_embeddings(args.embed_file))

            self.embed = nn.Embedding(W.size()[0], W.size()[1], padding_idx=0)
            self.embed.weight.data = W.clone()
        else:
            # add 2 to include UNK and PAD
            self.embed = nn.Embedding(len(dicts['w2ind']) + 2, args.embed_size, padding_idx=0)
        self.feature_size = self.embed.embedding_dim

        self.use_elmo = args.use_elmo
        if self.use_elmo:
            self.elmo = Elmo(args.elmo_options_file, args.elmo_weight_file, 1, requires_grad=args.elmo_tune,
                             dropout=args.elmo_dropout, gamma=args.elmo_gamma)
            with open(args.elmo_options_file, 'r') as fin:
                _options = json.load(fin)
            self.feature_size += _options['lstm']['projection_dim'] * 2

        self.embed_drop = nn.Dropout(p=args.dropout)

        self.conv_dict = {1: [self.feature_size, args.num_filter_maps],
                     2: [self.feature_size, 100, args.num_filter_maps],
                     3: [self.feature_size, 150, 100, args.num_filter_maps],
                     4: [self.feature_size, 200, 150, 100, args.num_filter_maps]
                     }


    def forward(self, x, target, text_inputs, use_elmo):

        features = [self.embed(x)]

        if use_elmo:
            elmo_outputs = self.elmo(text_inputs)
            elmo_outputs = elmo_outputs['elmo_representations'][0]
            features.append(elmo_outputs)

        x = torch.cat(features, dim=2)

        x = self.embed_drop(x)
        return x


class WordRepEH(nn.Module):
    def __init__(self, args, Y, dicts):
        super(WordRepEH, self).__init__()

        self.gpu = args.gpu

        if args.embed_file:
            print("loading pretrained embeddings from {}".format(args.embed_file))
            if args.use_ext_emb:
                pretrain_word_embedding, pretrain_emb_dim = build_pretrain_embedding(args.embed_file, dicts['w2ind'],
                                                                                     True)
                W = torch.from_numpy(pretrain_word_embedding)
            else:
                W = torch.Tensor(load_embeddings(args.embed_file))

            self.embed = nn.Embedding(W.size()[0], W.size()[1], padding_idx=0)
            self.embed.weight.data = W.clone()
        else:
            # add 2 to include UNK and PAD
            self.embed = nn.Embedding(len(dicts['w2ind']) + 2, args.embed_size, padding_idx=0)
        self.embed_entity = nn.Embedding(3, 8, padding_idx=0)
        self.embed_negation = nn.Embedding(3, 8, padding_idx=0)
        self.feature_size = self.embed.embedding_dim + 16

        self.use_elmo = args.use_elmo
        if self.use_elmo:
            self.elmo = Elmo(args.elmo_options_file, args.elmo_weight_file, 1, requires_grad=args.elmo_tune,
                             dropout=args.elmo_dropout, gamma=args.elmo_gamma)
            with open(args.elmo_options_file, 'r') as fin:
                _options = json.load(fin)
            self.feature_size += _options['lstm']['projection_dim'] * 2

        self.embed_drop = nn.Dropout(p=args.dropout)

        self.conv_dict = {1: [self.feature_size, args.num_filter_maps],
                     2: [self.feature_size, 100, args.num_filter_maps],
                     3: [self.feature_size, 150, 100, args.num_filter_maps],
                     4: [self.feature_size, 200, 150, 100, args.num_filter_maps]
                     }


    def forward(self, x, x_entity, x_negation, target, text_inputs, use_elmo):

        features = [self.embed(x), self.embed_entity(x_entity), self.embed_negation(x_negation)]

        if use_elmo:
            elmo_outputs = self.elmo(text_inputs)
            elmo_outputs = elmo_outputs['elmo_representations'][0]
            features.append(elmo_outputs)

        x = torch.cat(features, dim=2)

        x = self.embed_drop(x)
        return x

class OutputLayer(nn.Module):
    def __init__(self, args, Y, dicts, input_size):
        super(OutputLayer, self).__init__()

        self.U = nn.Linear(input_size, Y)
        xavier_uniform(self.U.weight)


        self.final = nn.Linear(input_size, Y)
        xavier_uniform(self.final.weight)

        self.loss_function = nn.BCEWithLogitsLoss()



    def forward(self, x, target, text_inputs):

        alpha = F.softmax(self.U.weight.matmul(x.transpose(1, 2)), dim=2)

        m = alpha.matmul(x)

        y = self.final.weight.mul(m).sum(dim=2).add(self.final.bias)

        loss = self.loss_function(y, target)
        return y, loss
from .resample_loss import ResampleLoss
class OutputLayerP(nn.Module):
    def __init__(self, args, Y, dicts, input_size):
        super(OutputLayerP, self).__init__()

        self.U = nn.Linear(input_size, Y)
        xavier_uniform(self.U.weight)
        self.gamma = args.gamma
        self.beta = args.beta
        self.lambd = args.lambd


        self.final = nn.Linear(input_size, Y)
        xavier_uniform(self.final.weight)

        self.loss_function = FocalLoss(class_frequency=dicts['freq_matrix'], gamma=self.gamma)



    def forward(self, x, target, text_inputs):

        alpha = F.softmax(self.U.weight.matmul(x.transpose(1, 2)), dim=2)

        m = alpha.matmul(x)

        y = self.final.weight.mul(m).sum(dim=2).add(self.final.bias)

        loss = self.loss_function(y, target)
        return y, loss

class OutputLayerProbI(nn.Module):
    def __init__(self, args, Y, dicts, input_size):
        super(OutputLayerProbI, self).__init__()

        self.U = nn.Linear(input_size, Y)
        xavier_uniform(self.U.weight)
        self.Y = Y

        self.final = nn.Linear(input_size, Y)
        xavier_uniform(self.final.weight)
        self.co_occur_matrix = nn.Parameter(torch.Tensor(dicts['prob_matrix']), requires_grad=args.train_distribution)
        self.att_cal = nn.Linear(input_size, 1)
        self.loss_function = nn.BCEWithLogitsLoss()



    def forward(self, x, target, text_inputs):

        alpha = F.softmax(self.U.weight.matmul(x.transpose(1, 2)), dim=2)

        m = alpha.matmul(x)

        y_ori = self.final.weight.mul(m).sum(dim=2).add(self.final.bias)
        y_adj = y_ori.matmul(self.co_occur_matrix)/self.Y
        att_adj = self.att_cal(m).squeeze()
        y = (1-att_adj)*y_ori + att_adj*y_adj
        loss = self.loss_function(y, target)
        return y, loss

class OutputLayerEntity(nn.Module):
    def __init__(self, args, Y, dicts, input_size):
        super(OutputLayerEntity, self).__init__()

        self.U = nn.Linear(input_size, Y)
        xavier_uniform(self.U.weight)


        self.final = nn.Linear(input_size, Y)
        xavier_uniform(self.final.weight)

        self.loss_function = nn.BCEWithLogitsLoss()



    def forward(self, x, x_entity, target, text_inputs):

        alpha = F.softmax(self.U.weight.matmul(x.transpose(1, 2)), dim=2)

        m = alpha.matmul(x)

        alpha2 = F.softmax(self.U.weight.matmul(x_entity.transpose(1, 2)), dim=2)

        m2 = alpha2.matmul(x_entity)

        y = self.final.weight.mul(m+m2).sum(dim=2).add(self.final.bias)

        loss = self.loss_function(y, target)
        return y, loss

class OutputLayerTF(nn.Module):
    def __init__(self, args, Y, dicts, input_size):
        super(OutputLayerTF, self).__init__()

        self.U = nn.Linear(input_size, Y)
        xavier_uniform(self.U.weight)

        self.U_gate = nn.Linear(input_size, Y)
        xavier_uniform(self.U_gate.weight)


        self.final = nn.Linear(input_size, Y)
        xavier_uniform(self.final.weight)

        self.gate_layer = nn.Linear(input_size, 1)

        self.loss_function = nn.BCEWithLogitsLoss()



    def forward(self, x, target, y_flow):

        alpha = F.softmax(self.U.weight.matmul(x.transpose(1, 2)), dim=2)

        m = alpha.matmul(x)

        alpha_gate = F.softmax(self.U_gate.weight.matmul(x.transpose(1, 2)), dim=2)

        m_gate = alpha_gate.matmul(x)
        gate = torch.tanh(self.gate_layer(m_gate).squeeze())

        y = self.final.weight.mul(m).sum(dim=2).add(self.final.bias) + y_flow * gate


        loss = self.loss_function(y, target)
        return y, loss

class OutputLayerMulti(nn.Module):
    def __init__(self, args, Y, dicts, input_size):
        super(OutputLayerMulti, self).__init__()
        self.ratio = args.hybrid_ratio.split(',')
        self.ratio = [int(x) for x in self.ratio]
        self.U_lv0 = nn.Linear(input_size, len(dicts['ind2c_lv0']))
        xavier_uniform(self.U_lv0.weight)
        self.U_lv1 = nn.Linear(input_size, len(dicts['ind2c_lv1']))
        xavier_uniform(self.U_lv1.weight)
        self.U_lv2 = nn.Linear(input_size, Y)
        xavier_uniform(self.U_lv2.weight)

        self.att_cal = nn.Linear(input_size, 1)

        self.final_lv0 = nn.Linear(input_size, len(dicts['ind2c_lv0']))
        self.lv02lv1 = torch.Tensor(dicts['lv02lv1']).cuda(args.gpu)
        xavier_uniform(self.final_lv0.weight)

        self.final_lv1 = nn.Linear(input_size, len(dicts['ind2c_lv1']))
        self.lv12lv2 = torch.Tensor(dicts['lv12lv2']).cuda(args.gpu)
        xavier_uniform(self.final_lv1.weight)

        self.final_lv2 = nn.Linear(input_size, Y)
        xavier_uniform(self.final_lv2.weight)

        self.layerW_lv0 = nn.Linear(input_size, input_size)
        self.layerW_lv1 = nn.Linear(input_size, 2*input_size)
        self.layerW_lv2 = nn.Linear(input_size, 2*input_size)
        self.loss_function = nn.BCEWithLogitsLoss()



    def forward(self, x, target_lv0, target_lv1, target_lv2, text_inputs):
        x_lv0 = self.layerW_lv0(x)
        lv0_emb = self.U_lv0.weight
        alpha_lv0 = F.softmax(lv0_emb.matmul(x_lv0.transpose(1, 2)), dim=2)
        m_lv0 = alpha_lv0.matmul(x_lv0)

        x_lv1 = self.layerW_lv1(x)
        lv1_bias = self.lv02lv1.T.matmul(lv0_emb)
        att = torch.sigmoid(self.att_cal(self.U_lv1.weight))
        lv1_hb = att * self.U_lv1.weight + (1 - att)*lv1_bias
        lv1_emb = torch.cat([lv1_hb, self.U_lv1.weight], 1)
        alpha_lv1 = F.softmax(lv1_emb.matmul(x_lv1.transpose(1, 2)), dim=2)
        m_lv1 = alpha_lv1.matmul(x_lv1)

        x_lv2 = self.layerW_lv2(x)
        lv2_bias = self.lv12lv2.T.matmul(lv1_hb)
        att = torch.sigmoid(self.att_cal(self.U_lv2.weight))
        lv2_hb = att * self.U_lv2.weight + (1 - att)*lv2_bias
        lv2_emb = torch.cat([lv2_hb, self.U_lv2.weight], 1)
        alpha_lv2 = F.softmax(lv2_emb.matmul(x_lv2.transpose(1, 2)), dim=2)
        m_lv2 = alpha_lv2.matmul(x_lv2)

        lv0_emb = self.final_lv0.weight
        y0 = lv0_emb.mul(m_lv0).sum(dim=2).add(self.final_lv0.bias)

        lv1_bias = self.lv02lv1.T.matmul(lv0_emb)
        att = torch.sigmoid(self.att_cal(self.final_lv1.weight))
        lv1_hb = att * self.final_lv1.weight + (1 - att) * lv1_bias
        lv1_emb = torch.cat([lv1_hb, self.final_lv1.weight], 1)
        y1 = lv1_emb.mul(m_lv1).sum(dim=2).add(self.final_lv1.bias)

        lv2_bias = self.lv12lv2.T.matmul(lv1_hb)
        att = torch.sigmoid(self.att_cal(self.final_lv2.weight))
        lv2_hb = att * self.final_lv2.weight + (1 - att) * lv2_bias
        lv2_emb = torch.cat([lv2_hb, self.final_lv2.weight], 1)
        y2 = lv2_emb.mul(m_lv2).sum(dim=2).add(self.final_lv2.bias)

        loss_v0 = self.loss_function(y0, target_lv0)
        loss_v1 = self.loss_function(y1, target_lv1)
        loss_v2 = self.loss_function(y2, target_lv2)
        loss = (self.ratio[0]*loss_v0 + self.ratio[1]*loss_v1 + self.ratio[2]*loss_v2) / sum(self.ratio)
        return y0, y1, y2, loss

class OutputLayerMultiRe(nn.Module):
    def __init__(self, args, Y, dicts, input_size):
        super(OutputLayerMultiRe, self).__init__()
        self.ratio = args.hybrid_ratio.split(',')
        self.ratio = [int(x) for x in self.ratio]
        self.U_lv0 = nn.Linear(input_size, len(dicts['ind2c_lv0']))
        xavier_uniform(self.U_lv0.weight)
        self.U_lv1 = nn.Linear(input_size, len(dicts['ind2c_lv1']))
        xavier_uniform(self.U_lv1.weight)
        self.U_lv2 = nn.Linear(input_size, Y)
        xavier_uniform(self.U_lv2.weight)

        self.att_cal = nn.Linear(2 * input_size, 1)

        self.final_lv0 = nn.Linear(input_size, len(dicts['ind2c_lv0']))
        self.lv02lv1 = torch.Tensor(dicts['lv02lv1']).cuda(args.gpu)
        xavier_uniform(self.final_lv0.weight)

        self.final_lv1 = nn.Linear(input_size, len(dicts['ind2c_lv1']))
        self.lv12lv2 = torch.Tensor(dicts['lv12lv2']).cuda(args.gpu)
        xavier_uniform(self.final_lv1.weight)

        self.final_lv2 = nn.Linear(input_size, Y)
        xavier_uniform(self.final_lv2.weight)

        self.layerW_lv0 = nn.Linear(input_size, input_size)
        self.layerW_lv1 = nn.Linear(input_size, 2*input_size)
        self.layerW_lv2 = nn.Linear(input_size, 2*input_size)
        self.loss_function = nn.BCEWithLogitsLoss()



    def forward(self, x, target_lv0, target_lv1, target_lv2, text_inputs):
        x_lv0 = self.layerW_lv0(x)
        lv0_emb = self.U_lv0.weight
        alpha_lv0 = F.softmax(lv0_emb.matmul(x_lv0.transpose(1, 2)), dim=2)
        m_lv0 = alpha_lv0.matmul(x_lv0)

        x_lv1 = self.layerW_lv1(x)
        lv1_bias = self.lv02lv1.T.matmul(lv0_emb)
        att = torch.sigmoid(self.att_cal(torch.cat([lv1_bias, self.U_lv1.weight], 1)))
        lv1_hb = att * self.U_lv1.weight + (1 - att)*lv1_bias
        lv1_emb = torch.cat([lv1_hb, self.U_lv1.weight], 1)
        alpha_lv1 = F.softmax(lv1_emb.matmul(x_lv1.transpose(1, 2)), dim=2)
        m_lv1 = alpha_lv1.matmul(x_lv1)

        x_lv2 = self.layerW_lv2(x)
        lv2_bias = self.lv12lv2.T.matmul(lv1_hb)
        att = torch.sigmoid(self.att_cal(torch.cat([lv2_bias, self.U_lv2.weight], 1)))
        lv2_hb = att * self.U_lv2.weight + (1 - att)*lv2_bias
        lv2_emb = torch.cat([lv2_hb, self.U_lv2.weight], 1)
        alpha_lv2 = F.softmax(lv2_emb.matmul(x_lv2.transpose(1, 2)), dim=2)
        m_lv2 = alpha_lv2.matmul(x_lv2)

        lv0_emb = self.final_lv0.weight
        y0 = lv0_emb.mul(m_lv0).sum(dim=2).add(self.final_lv0.bias)

        lv1_bias = self.lv02lv1.T.matmul(lv0_emb)
        att = torch.sigmoid(self.att_cal(torch.cat([lv1_bias, self.final_lv1.weight], 1)))
        lv1_hb = att * self.final_lv1.weight + (1 - att) * lv1_bias
        lv1_emb = torch.cat([lv1_hb, self.final_lv1.weight], 1)
        y1 = lv1_emb.mul(m_lv1).sum(dim=2).add(self.final_lv1.bias)

        lv2_bias = self.lv12lv2.T.matmul(lv1_hb)
        att = torch.sigmoid(self.att_cal(torch.cat([lv2_bias, self.final_lv2.weight], 1)))
        lv2_hb = att * self.final_lv2.weight + (1 - att) * lv2_bias
        lv2_emb = torch.cat([lv2_hb, self.final_lv2.weight], 1)
        y2 = lv2_emb.mul(m_lv2).sum(dim=2).add(self.final_lv2.bias)

        loss_v0 = self.loss_function(y0, target_lv0)
        loss_v1 = self.loss_function(y1, target_lv1)
        loss_v2 = self.loss_function(y2, target_lv2)
        loss = (self.ratio[0]*loss_v0 + self.ratio[1]*loss_v1 + self.ratio[2]*loss_v2) / sum(self.ratio)
        return y0, y1, y2, loss

class OutputLayerPro(nn.Module):
    def __init__(self, args, Y, dicts, input_size):
        super(OutputLayerPro, self).__init__()
        self.ratio = args.hybrid_ratio.split(',')
        self.lv2blank = torch.Tensor(dicts['lv2blank']).cuda(args.gpu)
        self.ratio = [int(x) for x in self.ratio]
        self.U_lv0 = nn.Linear(input_size, len(dicts['ind2c_lv0']))
        xavier_uniform(self.U_lv0.weight)
        self.U_lv1 = nn.Linear(input_size, len(dicts['ind2c_lv1']))
        xavier_uniform(self.U_lv1.weight)
        self.U_lv2 = nn.Linear(input_size, Y)
        xavier_uniform(self.U_lv2.weight)

        self.final_lv0 = nn.Linear(input_size, len(dicts['ind2c_lv0']))
        self.lv02lv1 = torch.Tensor(dicts['lv02lv1']).cuda(args.gpu)
        xavier_uniform(self.final_lv0.weight)

        self.final_lv1 = nn.Linear(input_size, len(dicts['ind2c_lv1']))
        self.lv12lv2 = torch.Tensor(dicts['lv12lv2']).cuda(args.gpu)
        xavier_uniform(self.final_lv1.weight)

        self.final_lv2 = nn.Linear(input_size, Y)
        xavier_uniform(self.final_lv2.weight)

        self.att_layer = nn.Linear(input_size, 3)
        self.loss_function = nn.BCEWithLogitsLoss()



    def forward(self, x, target_lv0, target_lv1, target_lv2, text_inputs):
        lv0_emb = self.U_lv0.weight
        alpha_lv0 = F.softmax(lv0_emb.matmul(x.transpose(1, 2)), dim=2)
        m_lv0 = alpha_lv0.matmul(x)

        lv1_emb = self.U_lv1.weight
        alpha_lv1 = F.softmax(lv1_emb.matmul(x.transpose(1, 2)), dim=2)
        m_lv1 = alpha_lv1.matmul(x)

        lv2_emb = self.U_lv2.weight
        alpha_lv2 = F.softmax(lv2_emb.matmul(x.transpose(1, 2)), dim=2)
        m_lv2 = alpha_lv2.matmul(x)

        lv0_emb = self.final_lv0.weight
        y0_ = lv0_emb.mul(m_lv0).sum(dim=2).add(self.final_lv0.bias)
        y0 = y0_.matmul(self.lv02lv1).matmul(self.lv12lv2)

        lv1_emb = self.final_lv1.weight
        y1_ = lv1_emb.mul(m_lv1).sum(dim=2).add(self.final_lv1.bias)
        y1 = y1_.matmul(self.lv12lv2)

        lv2_emb = self.final_lv2.weight
        y2 = lv2_emb.mul(m_lv2).sum(dim=2).add(self.final_lv2.bias)

        att = torch.softmax(self.att_layer(self.final_lv2.weight), dim=1)
        y = att[:, 0]*y0+att[:, 1]*y1+att[:, 2]*y2
        loss = self.loss_function(y, target_lv2)
        return y0_, y1_, y, loss

class OutputLayerProTg(nn.Module):
    def __init__(self, args, Y, dicts, input_size):
        super(OutputLayerProTg, self).__init__()
        self.ratio = args.hybrid_ratio.split(',')
        self.lv2blank = torch.Tensor(dicts['lv2blank']).cuda(args.gpu)
        self.ratio = [int(x) for x in self.ratio]
        self.U_lv0 = nn.Linear(input_size, len(dicts['ind2c_lv0']))
        xavier_uniform(self.U_lv0.weight)
        self.U_lv1 = nn.Linear(input_size, len(dicts['ind2c_lv1']))
        xavier_uniform(self.U_lv1.weight)
        self.U_lv2 = nn.Linear(input_size, Y)
        xavier_uniform(self.U_lv2.weight)

        self.final_lv0 = nn.Linear(input_size, len(dicts['ind2c_lv0']))
        self.lv02lv1 = torch.Tensor(dicts['lv02lv1']).cuda(args.gpu)
        xavier_uniform(self.final_lv0.weight)

        self.final_lv1 = nn.Linear(input_size, len(dicts['ind2c_lv1']))
        self.lv12lv2 = torch.Tensor(dicts['lv12lv2']).cuda(args.gpu)
        xavier_uniform(self.final_lv1.weight)

        self.final_lv2 = nn.Linear(input_size, Y)
        xavier_uniform(self.final_lv2.weight)

        self.layerW_lv0 = nn.Linear(input_size, input_size)
        self.layerW_lv1 = nn.Linear(input_size, input_size)
        self.layerW_lv2 = nn.Linear(input_size, input_size)
        self.loss_function = nn.BCEWithLogitsLoss()



    def forward(self, x, target_lv0, target_lv1, target_lv2, text_inputs):
        x_lv0 = self.layerW_lv0(x)
        lv0_emb = self.U_lv0.weight
        alpha_lv0 = F.softmax(lv0_emb.matmul(x_lv0.transpose(1, 2)), dim=2)
        m_lv0 = alpha_lv0.matmul(x_lv0)

        x_lv1 = self.layerW_lv1(x)
        lv1_emb = self.U_lv1.weight
        alpha_lv1 = F.softmax(lv1_emb.matmul(x_lv1.transpose(1, 2)), dim=2)
        m_lv1 = alpha_lv1.matmul(x_lv1)

        x_lv2 = self.layerW_lv2(x)
        lv2_emb = self.U_lv2.weight
        alpha_lv2 = F.softmax(lv2_emb.matmul(x_lv2.transpose(1, 2)), dim=2)
        m_lv2 = alpha_lv2.matmul(x_lv2)

        lv0_emb = self.final_lv0.weight
        y0 = lv0_emb.mul(m_lv0).sum(dim=2).add(self.final_lv0.bias)

        lv1_emb = self.final_lv1.weight
        lv1_bias = torch.sigmoid(y0).matmul(self.lv02lv1)
        y1 = lv1_emb.mul(m_lv1).sum(dim=2).add(self.final_lv1.bias)
        y1_ = (y1 + 1) * lv1_bias - 1

        lv2_emb = self.final_lv2.weight
        lv2_bias = torch.sigmoid(y1).matmul(self.lv12lv2)
        y2 = lv2_emb.mul(m_lv2).sum(dim=2).add(self.final_lv2.bias)
        y2_ = ((y2 + 1) * lv2_bias + (y2 + 1) * self.lv2blank) - 1

        loss_v0 = self.loss_function(y0, target_lv0)
        loss_v1 = self.loss_function(y1_, target_lv1)
        loss_v2 = self.loss_function(y2_, target_lv2)
        loss = (self.ratio[0]*loss_v0 + self.ratio[1]*loss_v1 + self.ratio[2]*loss_v2) / sum(self.ratio)
        return y0, y1_, y2_, loss

class OutputLayerMultiBase(nn.Module):
    def __init__(self, args, Y, dicts, input_size):
        super(OutputLayerMultiBase, self).__init__()

        self.U_lv0 = nn.Linear(input_size, len(dicts['ind2c_lv0']))
        xavier_uniform(self.U_lv0.weight)
        self.U_lv1 = nn.Linear(input_size, len(dicts['ind2c_lv1']))
        xavier_uniform(self.U_lv1.weight)
        self.U_lv2 = nn.Linear(input_size, Y)
        xavier_uniform(self.U_lv2.weight)


        self.final_lv0 = nn.Linear(input_size, len(dicts['ind2c_lv0']))
        self.lv02lv1 = torch.Tensor(dicts['lv02lv1']).cuda(args.gpu)
        xavier_uniform(self.final_lv0.weight)

        self.final_lv1 = nn.Linear(input_size, len(dicts['ind2c_lv1']))
        self.lv12lv2 = torch.Tensor(dicts['lv12lv2']).cuda(args.gpu)
        xavier_uniform(self.final_lv1.weight)

        self.final_lv2 = nn.Linear(input_size, Y)
        xavier_uniform(self.final_lv2.weight)

        self.loss_function = nn.BCEWithLogitsLoss()



    def forward(self, x, target_lv0, target_lv1, target_lv2, text_inputs):
        lv0_emb = self.U_lv0.weight
        alpha_lv0 = F.softmax(lv0_emb.matmul(x.transpose(1, 2)), dim=2)
        m_lv0 = alpha_lv0.matmul(x)

        lv1_bias = self.lv02lv1.T.matmul(lv0_emb)
        lv1_emb = self.U_lv1.weight + lv1_bias
        alpha_lv1 = F.softmax(lv1_emb.matmul(x.transpose(1, 2)), dim=2)
        m_lv1 = alpha_lv1.matmul(x)

        lv2_bias = self.lv12lv2.T.matmul(lv1_emb)
        lv2_emb = self.U_lv2.weight + lv2_bias
        alpha_lv2 = F.softmax(lv2_emb.matmul(x.transpose(1, 2)), dim=2)
        m_lv2 = alpha_lv2.matmul(x)

        lv0_emb = self.final_lv0.weight
        y0 = lv0_emb.mul(m_lv0).sum(dim=2).add(self.final_lv0.bias)

        lv1_bias = self.lv02lv1.T.matmul(lv0_emb)
        lv1_emb = self.final_lv1.weight + lv1_bias
        y1 = lv1_emb.mul(m_lv1).sum(dim=2).add(self.final_lv1.bias)

        lv2_bias = self.lv12lv2.T.matmul(lv1_emb)
        lv2_emb = self.final_lv2.weight + lv2_bias
        y2 = lv2_emb.mul(m_lv2).sum(dim=2).add(self.final_lv2.bias)

        loss_v0 = self.loss_function(y0, target_lv0)
        loss_v1 = self.loss_function(y1, target_lv1)
        loss_v2 = self.loss_function(y2, target_lv2)
        loss = (loss_v0 + loss_v1 + loss_v2) / 3
        return y2, loss

class OutputLayerMultiNaive(nn.Module):
    def __init__(self, args, Y, dicts, input_size):
        super(OutputLayerMultiNaive, self).__init__()

        self.U_lv0 = nn.Linear(input_size, len(dicts['ind2c_lv0']))
        xavier_uniform(self.U_lv0.weight)
        self.U_lv1 = nn.Linear(input_size, len(dicts['ind2c_lv1']))
        xavier_uniform(self.U_lv1.weight)
        self.U_lv2 = nn.Linear(input_size, Y)
        xavier_uniform(self.U_lv2.weight)


        self.final_lv0 = nn.Linear(input_size, len(dicts['ind2c_lv0']))
        self.lv02lv1 = torch.Tensor(dicts['lv02lv1']).cuda(args.gpu)
        xavier_uniform(self.final_lv0.weight)

        self.final_lv1 = nn.Linear(input_size, len(dicts['ind2c_lv1']))
        self.lv12lv2 = torch.Tensor(dicts['lv12lv2']).cuda(args.gpu)
        xavier_uniform(self.final_lv1.weight)

        self.final_lv2 = nn.Linear(input_size, Y)
        xavier_uniform(self.final_lv2.weight)

        self.loss_function = nn.BCEWithLogitsLoss()



    def forward(self, x, target_lv0, target_lv1, target_lv2, text_inputs):
        lv0_emb = self.U_lv0.weight
        alpha_lv0 = F.softmax(lv0_emb.matmul(x.transpose(1, 2)), dim=2)
        m_lv0 = alpha_lv0.matmul(x)

        lv1_emb = self.U_lv1.weight
        alpha_lv1 = F.softmax(lv1_emb.matmul(x.transpose(1, 2)), dim=2)
        m_lv1 = alpha_lv1.matmul(x)

        lv2_emb = self.U_lv2.weight
        alpha_lv2 = F.softmax(lv2_emb.matmul(x.transpose(1, 2)), dim=2)
        m_lv2 = alpha_lv2.matmul(x)

        lv0_emb = self.final_lv0.weight
        y0 = lv0_emb.mul(m_lv0).sum(dim=2).add(self.final_lv0.bias)

        lv1_emb = self.final_lv1.weight
        y1 = lv1_emb.mul(m_lv1).sum(dim=2).add(self.final_lv1.bias)

        lv2_emb = self.final_lv2.weight
        y2 = lv2_emb.mul(m_lv2).sum(dim=2).add(self.final_lv2.bias)

        loss_v0 = self.loss_function(y0, target_lv0)
        loss_v1 = self.loss_function(y1, target_lv1)
        loss_v2 = self.loss_function(y2, target_lv2)
        loss = (loss_v0 + loss_v1 + loss_v2) / 3
        return y2, loss

class CNN(nn.Module):
    def __init__(self, args, Y, dicts):
        super(CNN, self).__init__()

        self.word_rep = WordRep(args, Y, dicts)

        filter_size = int(args.filter_size)


        self.conv = nn.Conv1d(self.word_rep.feature_size, args.num_filter_maps, kernel_size=filter_size,
                                  padding=int(floor(filter_size / 2)))
        xavier_uniform(self.conv.weight)

        self.output_layer = OutputLayer(args, Y, dicts, args.num_filter_maps)


    def forward(self, x, target, text_inputs):

        x = self.word_rep(x, target, text_inputs)

        x = x.transpose(1, 2)

        x = torch.tanh(self.conv(x).transpose(1, 2))

        y, loss = self.output_layer(x, target, text_inputs)
        return y, loss

    def freeze_net(self):
        for p in self.word_rep.embed.parameters():
            p.requires_grad = False


class MultiCNN(nn.Module):
    def __init__(self, args, Y, dicts):
        super(MultiCNN, self).__init__()

        self.word_rep = WordRep(args, Y, dicts)

        if args.filter_size.find(',') == -1:
            self.filter_num = 1
            filter_size = int(args.filter_size)
            self.conv = nn.Conv1d(self.word_rep.feature_size, args.num_filter_maps, kernel_size=filter_size,
                                  padding=int(floor(filter_size / 2)))
            xavier_uniform(self.conv.weight)
        else:
            filter_sizes = args.filter_size.split(',')
            self.filter_num = len(filter_sizes)
            self.conv = nn.ModuleList()
            for filter_size in filter_sizes:
                filter_size = int(filter_size)
                tmp = nn.Conv1d(self.word_rep.feature_size, args.num_filter_maps, kernel_size=filter_size,
                                      padding=int(floor(filter_size / 2)))
                xavier_uniform(tmp.weight)
                self.conv.add_module('conv-{}'.format(filter_size), tmp)

        self.output_layer = OutputLayer(args, Y, dicts, self.filter_num * args.num_filter_maps)



    def forward(self, x, target, text_inputs):

        x = self.word_rep(x, target, text_inputs)

        x = x.transpose(1, 2)

        if self.filter_num == 1:
            x = torch.tanh(self.conv(x).transpose(1, 2))
        else:
            conv_result = []
            for tmp in self.conv:
                conv_result.append(torch.tanh(tmp(x).transpose(1, 2)))
            x = torch.cat(conv_result, dim=2)

        y, loss = self.output_layer(x, target, text_inputs)

        return y, loss

    def freeze_net(self):
        for p in self.word_rep.embed.parameters():
            p.requires_grad = False


class ResidualBlock(nn.Module):
    def __init__(self, inchannel, outchannel, kernel_size, stride, use_res, dropout):
        super(ResidualBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv1d(inchannel, outchannel, kernel_size=kernel_size, stride=stride, padding=int(floor(kernel_size / 2)), bias=False),
            nn.BatchNorm1d(outchannel),
            nn.Tanh(),
            nn.Conv1d(outchannel, outchannel, kernel_size=kernel_size, stride=1, padding=int(floor(kernel_size / 2)), bias=False),
            nn.BatchNorm1d(outchannel)
        )

        self.use_res = use_res
        if self.use_res:
            self.shortcut = nn.Sequential(
                        nn.Conv1d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False),
                        nn.BatchNorm1d(outchannel)
                    )

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        out = self.left(x)
        if self.use_res:
            out += self.shortcut(x)
        out = torch.tanh(out)
        out = self.dropout(out)
        return out

class ResidualBlockHidden(nn.Module):
    def __init__(self, inchannel, outchannel, kernel_size, stride, use_res, dropout, use_layer_norm=False, is_relu=True):
        super(ResidualBlockHidden, self).__init__()
        self.left = nn.Sequential(
            nn.Conv1d(inchannel, outchannel, kernel_size=kernel_size, stride=stride, padding=int(floor(kernel_size / 2)), bias=False),
            nn.GroupNorm(1, outchannel) if use_layer_norm else nn.BatchNorm1d(outchannel),
            nn.Tanh() if not is_relu else nn.LeakyReLU(),
            nn.Conv1d(outchannel, outchannel, kernel_size=kernel_size, stride=1, padding=int(floor(kernel_size / 2)), bias=False),
            nn.GroupNorm(1, outchannel) if use_layer_norm else nn.BatchNorm1d(outchannel),
        )

        self.use_res = use_res
        if self.use_res:
            self.shortcut = nn.Sequential(
                        nn.Conv1d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False),
                        nn.GroupNorm(1, outchannel) if use_layer_norm else nn.BatchNorm1d(outchannel)
                    )
        self.dropout = nn.Dropout(p=dropout)
        self.out_activation = nn.Tanh() if not is_relu else nn.LeakyReLU()


    def forward(self, x):
        out = self.left(x)
        if self.use_res:
            out += self.shortcut(x)
        out = self.out_activation(out)
        out = self.dropout(out)
        return out

class AttentionBolck(nn.Module):
    def __init__(self, inchannel, pool_size, is_max_pool=True):
        super(AttentionBolck, self).__init__()
        self.is_max_pool = is_max_pool
        self.att_conv = nn.Sequential(
            nn.Conv1d(inchannel, 1, kernel_size=1, stride=1, bias=False),
            nn.Sigmoid()
        )
        self.squeeze_pool = nn.MaxPool1d(pool_size, pool_size, return_indices=True) if is_max_pool else nn.AvgPool1d(pool_size, pool_size)



    def forward(self, x):
        if self.is_max_pool:
            att = self.att_conv(x)
            x = x * att
            _, indices = self.squeeze_pool(att)
            indices = indices.expand(indices.shape[0], x.shape[1], indices.shape[2])
            x = torch.gather(x, 2, indices)
        else:
            att = self.att_conv(x)
            x = x * att
            x = self.squeeze_pool(x)
        return x

class AttentionBolckV2(nn.Module):
    def __init__(self, inchannel, pool_size, is_max_pool=True):
        super(AttentionBolckV2, self).__init__()
        self.is_max_pool = is_max_pool
        self.pool_size = pool_size
        self.att_conv = nn.Sequential(
            nn.Conv1d(inchannel, 1, kernel_size=1, stride=1, bias=False),
        )
        self.squeeze_pool = nn.MaxPool1d(pool_size, pool_size, return_indices=True) if is_max_pool else nn.AvgPool1d(pool_size, pool_size)



    def forward(self, x):
        if self.is_max_pool:
            if x.shape[2] % self.pool_size != 0:
                x = torch.nn.functional.pad(x, [0, (self.pool_size - (x.shape[2] % self.pool_size))])
            att = self.att_conv(x)
            att = att.view(att.shape[0], att.shape[1], -1, self.pool_size)
            att = torch.softmax(att, dim=3)
            x = x.view(x.shape[0], x.shape[1], -1, self.pool_size)
            x = x * att
            x = torch.sum(x, dim=3)
        else:
            att = self.att_conv(x)
            x = x * att
            x = self.squeeze_pool(x)
        return x

class AttentionOnly(nn.Module):
    def __init__(self, inchannel, pool_size, is_max_pool=True):
        super(AttentionOnly, self).__init__()
        self.is_max_pool = is_max_pool
        self.pool_size = pool_size
        self.att_conv = nn.Sequential(
            nn.Conv1d(inchannel, 1, kernel_size=1, stride=1, bias=False),
        )
        self.squeeze_pool = nn.MaxPool1d(pool_size, pool_size, return_indices=True) if is_max_pool else nn.AvgPool1d(pool_size, pool_size)
    def forward(self, x):
        att = self.att_conv(x)
        att = torch.sigmoid(att)
        x = x * att
        return x

class AttentionBolckV3(nn.Module):
    def __init__(self, inchannel, pool_size, is_max_pool=True, pow_n=5):
        super(AttentionBolckV3, self).__init__()
        self.is_max_pool = is_max_pool
        self.pow_n = pow_n
        self.pool_size = pool_size
        self.att_conv = nn.Sequential(
            nn.Conv1d(inchannel, 1, kernel_size=1, stride=1, bias=False),
        )
        self.squeeze_pool = nn.MaxPool1d(pool_size, pool_size, return_indices=True) if is_max_pool else nn.AvgPool1d(
            pool_size, pool_size)

    def forward(self, x):
        if self.is_max_pool:
            if x.shape[2] % 2 != 0:
                x = torch.nn.functional.pad(x, [0, 1])
            att = self.att_conv(x)
            att = att.view(att.shape[0], att.shape[1], -1, self.pool_size)
            att = torch.softmax(att, dim=3)
            delta = torch.max(att, dim=3, keepdim=True)[0]
            delta = 1+torch.pow(2-2*delta, self.pow_n)
            att = att * delta
            x = x.view(x.shape[0], x.shape[1], -1, self.pool_size)
            x = x * att
            x = torch.sum(x, dim=3)
        else:
            att = self.att_conv(x)
            x = x * att
            x = self.squeeze_pool(x)
        return x


class ResCNN(nn.Module):

    def __init__(self, args, Y, dicts):
        super(ResCNN, self).__init__()

        self.word_rep = WordRep(args, Y, dicts)

        self.conv = nn.ModuleList()
        conv_dimension = self.word_rep.conv_dict[args.conv_layer]
        for idx in range(args.conv_layer):
            tmp = ResidualBlock(conv_dimension[idx], conv_dimension[idx + 1], int(args.filter_size), 1, True, args.dropout)
            self.conv.add_module('conv-{}'.format(idx), tmp)

        self.output_layer = OutputLayer(args, Y, dicts, args.num_filter_maps)


    def forward(self, x, target, text_inputs):

        x = self.word_rep(x, target, text_inputs)

        x = x.transpose(1, 2)

        for conv in self.conv:
            x = conv(x)
        x = x.transpose(1, 2)

        y, loss = self.output_layer(x, target, text_inputs)

        return y, loss

    def freeze_net(self):
        for p in self.word_rep.embed.parameters():
            p.requires_grad = False


class MultiResCNN(nn.Module):

    def __init__(self, args, Y, dicts):
        super(MultiResCNN, self).__init__()

        self.word_rep = WordRep(args, Y, dicts)
        self.use_elmo = args.use_elmo
        self.conv = nn.ModuleList()
        filter_sizes = args.filter_size.split(',')

        self.filter_num = len(filter_sizes)
        for filter_size in filter_sizes:
            filter_size = int(filter_size)
            one_channel = nn.ModuleList()
            tmp = nn.Conv1d(self.word_rep.feature_size, self.word_rep.feature_size, kernel_size=filter_size,
                            padding=int(floor(filter_size / 2)))
            xavier_uniform(tmp.weight)
            one_channel.add_module('baseconv', tmp)

            conv_dimension = self.word_rep.conv_dict[args.conv_layer]
            for idx in range(args.conv_layer):
                tmp = ResidualBlock(conv_dimension[idx], conv_dimension[idx + 1], filter_size, 1, True,
                                    args.dropout)
                one_channel.add_module('resconv-{}'.format(idx), tmp)

            self.conv.add_module('channel-{}'.format(filter_size), one_channel)

        self.output_layer = OutputLayer(args, Y, dicts, self.filter_num * args.num_filter_maps)


    def forward(self, x, target, text_inputs):

        x = self.word_rep(x, target, text_inputs, self.use_elmo)

        x = x.transpose(1, 2)

        conv_result = []
        for conv in self.conv:
            tmp = x
            for idx, md in enumerate(conv):
                if idx == 0:
                    tmp = torch.tanh(md(tmp))
                else:
                    tmp = md(tmp)
            tmp = tmp.transpose(1, 2)
            conv_result.append(tmp)
        x = torch.cat(conv_result, dim=2)

        y, loss = self.output_layer(x, target, text_inputs)

        return y, loss

    def freeze_net(self):
        for p in self.word_rep.embed.parameters():
            p.requires_grad = False

class MultiResCNNRNN(nn.Module):

    def __init__(self, args, Y, dicts):
        super(MultiResCNNRNN, self).__init__()

        self.word_rep = WordRep(args, Y, dicts)

        self.conv = nn.ModuleList()
        filter_sizes = args.filter_size.split(',')
        self.rnn = nn.GRU(args.rnn_dim, args.rnn_dim, 1,
                          bidirectional=False, batch_first=True)
        self.relu = nn.ReLU(inplace=True)
        self.embed_rnn_drop = nn.Dropout(p=args.dropout)
        self.use_elmo = args.use_elmo
        self.word_rep_rnn = nn.Embedding(Y + 2, args.rnn_dim, padding_idx=0)
        self.transfer_rnn = nn.Linear(args.rnn_dim, Y)
        #self.word_rep_entity = nn.Linear(args.entity_size, args.entity_dim)
        #self.word_rep_entity = nn.Embedding(args.entity_size + 2, args.entity_dim, padding_idx=0)
        #if args.entity_embedding:
        #    self.word_rep_entity.weight = np.load(args.entity_embedding)

        self.filter_num = len(filter_sizes)
        for filter_size in filter_sizes:
            filter_size = int(filter_size)
            one_channel = nn.ModuleList()
            tmp = nn.Conv1d(self.word_rep.feature_size, self.word_rep.feature_size, kernel_size=filter_size,
                            padding=int(floor(filter_size / 2)))
            xavier_uniform(tmp.weight)
            one_channel.add_module('baseconv', tmp)

            conv_dimension = self.word_rep.conv_dict[args.conv_layer]
            for idx in range(args.conv_layer):
                tmp = ResidualBlock(conv_dimension[idx], conv_dimension[idx + 1], filter_size, 1, True,
                                    args.dropout)
                one_channel.add_module('resconv-{}'.format(idx), tmp)

            self.conv.add_module('channel-{}'.format(filter_size), one_channel)

        self.output_layer = OutputLayerTF(args, Y, dicts, self.filter_num * args.num_filter_maps)
        #self.entity_gate = nn.Linear(self.filter_num * args.num_filter_maps, 1)


    def forward(self, x, target, text_inputs, p_codes, p_codes_final_mask):

        x = self.word_rep(x, None, text_inputs, self.use_elmo)

        x = x.transpose(1, 2)

        conv_result = []
        for conv in self.conv:
            tmp = x
            for idx, md in enumerate(conv):
                if idx == 0:
                    tmp = torch.tanh(md(tmp))
                else:
                    tmp = md(tmp)
            tmp = tmp.transpose(1, 2)
            conv_result.append(tmp)
        x = torch.cat(conv_result, dim=2)

        # apply Time Flow RNN
        p_codes = self.word_rep_rnn(p_codes).sum(dim=2)
        p_codes = self.embed_rnn_drop(p_codes)
        out, hidden = self.rnn(p_codes)
        out = (out * torch.unsqueeze(p_codes_final_mask, 2)).sum(dim=1)
        y_flow = self.transfer_rnn(out)
        y, loss = self.output_layer(x, target, y_flow)

        return y, loss

    def freeze_net(self):
        for p in self.word_rep.embed.parameters():
            p.requires_grad = False

class MultiResCNNEH(nn.Module):

    def __init__(self, args, Y, dicts):
        super(MultiResCNNEH, self).__init__()

        self.word_rep = WordRepEH(args, Y, dicts)

        self.conv = nn.ModuleList()
        filter_sizes = args.filter_size.split(',')
        self.relu = nn.ReLU(inplace=True)
        self.use_elmo = args.use_elmo
        self.use_transformer = args.use_transformer

        self.filter_num = len(filter_sizes)
        for filter_size in filter_sizes:
            filter_size = int(filter_size)
            one_channel = nn.ModuleList()
            tmp = nn.Conv1d(self.word_rep.feature_size, self.word_rep.feature_size, kernel_size=filter_size,
                            padding=int(floor(filter_size / 2)))
            xavier_uniform(tmp.weight)
            one_channel.add_module('baseconv', tmp)

            conv_dimension = self.word_rep.conv_dict[args.conv_layer]
            for idx in range(args.conv_layer):
                tmp = ResidualBlockHidden(conv_dimension[idx], conv_dimension[idx + 1], filter_size, 1, True, args.dropout if args.use_transformer else 0.0, use_layer_norm=args.use_layer_norm, is_relu=args.use_relu)
                one_channel.add_module('resconv-{}'.format(idx), tmp)

            self.conv.add_module('channel-{}'.format(filter_size), one_channel)

        self.output_layer = OutputLayer(args, Y, dicts, self.filter_num * args.num_filter_maps)
        if args.use_transformer:
            self.transfer = EncoderHidden(len(dicts['e2ind']), args.MAX_LENGTH, args.transfer_layer,
                                          self.filter_num * args.num_filter_maps, args.transfer_attention_head, 1024,
                                          args.dropout, gpu=args.gpu)


    def forward(self, x, x_entity, x_negation, target, text_inputs, p_codes, p_codes_final_mask):

        x = self.word_rep(x, x_entity, x_negation, None, text_inputs, self.use_elmo)

        x = x.transpose(1, 2)

        conv_result = []
        for conv in self.conv:
            tmp = x
            for idx, md in enumerate(conv):
                if idx == 0:
                    tmp = torch.tanh(md(tmp))
                else:
                    tmp = md(tmp)
            tmp = tmp.transpose(1, 2)
            conv_result.append(tmp)
        x = torch.cat(conv_result, dim=2)
        if self.use_transformer:
            x = self.transfer(x)

        y, loss = self.output_layer(x, target, None)

        return y, loss

    def freeze_net(self):
        for p in self.word_rep.embed.parameters():
            p.requires_grad = False


class MultiResCNNEntity(nn.Module):

    def __init__(self, args, Y, dicts):
        super(MultiResCNNEntity, self).__init__()

        self.word_rep = WordRepEH(args, Y, dicts)

        self.conv = nn.ModuleList()
        filter_sizes = args.filter_size.split(',')
        self.rnn = nn.GRU(args.rnn_dim, args.rnn_dim, 1,
                          bidirectional=False, batch_first=True)
        self.relu = nn.ReLU(inplace=True)
        self.use_elmo = args.use_elmo

        self.filter_num = len(filter_sizes)
        for filter_size in filter_sizes:
            filter_size = int(filter_size)
            one_channel = nn.ModuleList()
            tmp = nn.Conv1d(self.word_rep.feature_size, self.word_rep.feature_size, kernel_size=filter_size,
                            padding=int(floor(filter_size / 2)))
            xavier_uniform(tmp.weight)
            one_channel.add_module('baseconv', tmp)

            conv_dimension = self.word_rep.conv_dict[args.conv_layer]
            for idx in range(args.conv_layer):
                tmp = ResidualBlock(conv_dimension[idx], conv_dimension[idx + 1], filter_size, 1, True,
                                    args.dropout)
                one_channel.add_module('resconv-{}'.format(idx), tmp)

            self.conv.add_module('channel-{}'.format(filter_size), one_channel)

        self.output_layer = OutputLayerEntity(args, Y, dicts, self.filter_num * args.num_filter_maps)
        self.transfer = EncoderHidden(len(dicts['e2ind']), args.MAX_LENGTH, args.transfer_layer, self.filter_num * args.num_filter_maps, args.transfer_attention_head, 1024, args.dropout, gpu=args.gpu)


    def forward(self, x, x_entity, x_negation, x_entity_list, x_negation_list, target, text_inputs, p_codes, p_codes_final_mask):

        x = self.word_rep(x, x_entity, x_negation, None, text_inputs, self.use_elmo)

        x = x.transpose(1, 2)

        conv_result = []
        for conv in self.conv:
            tmp = x
            for idx, md in enumerate(conv):
                if idx == 0:
                    tmp = torch.tanh(md(tmp))
                else:
                    tmp = md(tmp)
            tmp = tmp.transpose(1, 2)
            conv_result.append(tmp)
        x = torch.cat(conv_result, dim=2)
        x_entity = self.transfer(x_entity_list, x_negation_list)
        y, loss = self.output_layer(x, x_entity, target, None)

        return y, loss

    def freeze_net(self):
        for p in self.word_rep.embed.parameters():
            p.requires_grad = False

class MultiResCNNEntityHidden(nn.Module):

    def __init__(self, args, Y, dicts):
        super(MultiResCNNEntityHidden, self).__init__()

        self.word_rep = WordRepEH(args, Y, dicts)

        self.conv = nn.ModuleList()
        filter_sizes = args.filter_size.split(',')
        self.relu = nn.ReLU(inplace=True)
        self.use_elmo = args.use_elmo
        self.use_transformer = args.use_transformer
        self.filter_num = len(filter_sizes)
        for filter_size in filter_sizes:
            filter_size = int(filter_size)
            one_channel = nn.ModuleList()
            tmp = nn.Conv1d(self.word_rep.feature_size, self.word_rep.feature_size, kernel_size=filter_size,
                            padding=int(floor(filter_size / 2)))
            xavier_uniform(tmp.weight)
            one_channel.add_module('baseconv', tmp)
            tmp = AttentionBolckV2(self.word_rep.feature_size, args.pool_size, True)
            one_channel.add_module('basevonb-pool', tmp)

            conv_dimension = self.word_rep.conv_dict[args.conv_layer]
            for idx in range(args.conv_layer):
                tmp = ResidualBlockHidden(conv_dimension[idx], conv_dimension[idx + 1], filter_size, 1, True, args.dropout if args.use_transformer else 0.0, use_layer_norm=args.use_layer_norm, is_relu=args.use_relu)
                one_channel.add_module('resconv-{}'.format(idx), tmp)
            self.conv.add_module('channel-{}'.format(filter_size), one_channel)

        self.output_layer = OutputLayer(args, Y, dicts, self.filter_num * args.num_filter_maps)
        if args.use_transformer:
            self.transfer = EncoderHidden(len(dicts['e2ind']), args.MAX_LENGTH, args.transfer_layer, self.filter_num * args.num_filter_maps, args.transfer_attention_head, args.transfer_fsize, args.dropout, gpu=args.gpu)
#
    def forward(self, x, x_entity, x_negation, target, text_inputs, p_codes, p_codes_final_mask):

        x = self.word_rep(x, x_entity, x_negation, None, text_inputs, self.use_elmo)

        x = x.transpose(1, 2)

        conv_result = []
        for conv in self.conv:
            tmp = x
            for idx, md in enumerate(conv):
                if idx == 0:
                    tmp = torch.tanh(md(tmp))
                else:
                    tmp = md(tmp)
            tmp = tmp.transpose(1, 2)
            conv_result.append(tmp)
        x = torch.cat(conv_result, dim=2)
        if self.use_transformer:
            x = self.transfer(x)
        y, loss = self.output_layer(x, target, None)

        return y, loss

    def freeze_net(self):
        for p in self.word_rep.embed.parameters():
            p.requires_grad = False

class MultiResCNNHidden(nn.Module):

    def __init__(self, args, Y, dicts):
        super(MultiResCNNHidden, self).__init__()

        self.word_rep = WordRep(args, Y, dicts)

        self.conv = nn.ModuleList()
        filter_sizes = args.filter_size.split(',')
        self.relu = nn.ReLU(inplace=True)
        self.use_elmo = args.use_elmo
        self.use_transformer = args.use_transformer
        self.filter_num = len(filter_sizes)
        for filter_size in filter_sizes:
            filter_size = int(filter_size)
            one_channel = nn.ModuleList()
            tmp = nn.Conv1d(self.word_rep.feature_size, self.word_rep.feature_size, kernel_size=filter_size,
                            padding=int(floor(filter_size / 2)))
            xavier_uniform(tmp.weight)
            one_channel.add_module('baseconv', tmp)
            if args.use_attention_pool:
                for idx in args.compressor_layer:
                    tmp = AttentionBolckV2(self.word_rep.feature_size, args.pool_size, True)
                    one_channel.add_module('basevonb-pool-{}'.format(idx), tmp)

            conv_dimension = self.word_rep.conv_dict[args.conv_layer]
            for idx in range(args.conv_layer):
                tmp = ResidualBlockHidden(conv_dimension[idx], conv_dimension[idx + 1], filter_size, 1, True, args.dropout if args.use_transformer else 0.0, use_layer_norm=args.use_layer_norm, is_relu=args.use_relu)
                one_channel.add_module('resconv-{}'.format(idx), tmp)
            self.conv.add_module('channel-{}'.format(filter_size), one_channel)

        self.output_layer = OutputLayer(args, Y, dicts, self.filter_num * args.num_filter_maps)
        if args.use_transformer:
            self.transfer = EncoderHidden(len(dicts['e2ind']), args.MAX_LENGTH, args.transfer_layer, self.filter_num * args.num_filter_maps, args.transfer_attention_head, args.transfer_fsize, args.dropout, gpu=args.gpu)
#
    def forward(self, x, target, text_inputs):

        x = self.word_rep(x, None, text_inputs, self.use_elmo)

        x = x.transpose(1, 2)

        conv_result = []
        for conv in self.conv:
            tmp = x
            for idx, md in enumerate(conv):
                if idx == 0:
                    tmp = torch.tanh(md(tmp))
                else:
                    tmp = md(tmp)
            tmp = tmp.transpose(1, 2)
            conv_result.append(tmp)
        x = torch.cat(conv_result, dim=2)
        if self.use_transformer:
            x = self.transfer(x)
        y, loss = self.output_layer(x, target, None)

        return y, loss

    def freeze_net(self):
        for p in self.word_rep.embed.parameters():
            p.requires_grad = False

class MultiResCNNHiddenProbI(nn.Module):

    def __init__(self, args, Y, dicts):
        super(MultiResCNNHiddenProbI, self).__init__()

        self.word_rep = WordRep(args, Y, dicts)

        self.conv = nn.ModuleList()
        filter_sizes = args.filter_size.split(',')
        self.relu = nn.ReLU(inplace=True)
        self.use_elmo = args.use_elmo
        self.use_transformer = args.use_transformer
        self.filter_num = len(filter_sizes)
        for filter_size in filter_sizes:
            filter_size = int(filter_size)
            one_channel = nn.ModuleList()
            tmp = nn.Conv1d(self.word_rep.feature_size, self.word_rep.feature_size, kernel_size=filter_size,
                            padding=int(floor(filter_size / 2)))
            xavier_uniform(tmp.weight)
            one_channel.add_module('baseconv', tmp)
            tmp = AttentionBolckV2(self.word_rep.feature_size, args.pool_size, True)
            one_channel.add_module('basevonb-pool', tmp)

            conv_dimension = self.word_rep.conv_dict[args.conv_layer]
            for idx in range(args.conv_layer):
                tmp = ResidualBlockHidden(conv_dimension[idx], conv_dimension[idx + 1], filter_size, 1, True, args.dropout if args.use_transformer else 0.0, use_layer_norm=args.use_layer_norm, is_relu=args.use_relu)
                one_channel.add_module('resconv-{}'.format(idx), tmp)
            self.conv.add_module('channel-{}'.format(filter_size), one_channel)

        self.output_layer = OutputLayerP(args, Y, dicts, self.filter_num * args.num_filter_maps)
        if args.use_transformer:
            self.transfer = EncoderHidden(len(dicts['e2ind']), args.MAX_LENGTH, args.transfer_layer, self.filter_num * args.num_filter_maps, args.transfer_attention_head, args.transfer_fsize, args.dropout, gpu=args.gpu)
#
    def forward(self, x, target, text_inputs):

        x = self.word_rep(x, None, text_inputs, self.use_elmo)

        x = x.transpose(1, 2)

        conv_result = []
        for conv in self.conv:
            tmp = x
            for idx, md in enumerate(conv):
                if idx == 0:
                    tmp = torch.tanh(md(tmp))
                else:
                    tmp = md(tmp)
            tmp = tmp.transpose(1, 2)
            conv_result.append(tmp)
        x = torch.cat(conv_result, dim=2)
        if self.use_transformer:
            x = self.transfer(x)
        y, loss = self.output_layer(x, target, None)

        return y, loss

    def freeze_net(self):
        for p in self.word_rep.embed.parameters():
            p.requires_grad = False

class MultiResCNNHiddenHybrid(nn.Module):

    def __init__(self, args, Y, dicts):
        super(MultiResCNNHiddenHybrid, self).__init__()

        self.word_rep = WordRep(args, Y, dicts)

        self.conv = nn.ModuleList()
        filter_sizes = args.filter_size.split(',')
        self.relu = nn.ReLU(inplace=True)
        self.use_elmo = args.use_elmo
        self.use_transformer = args.use_transformer
        self.filter_num = len(filter_sizes)
        for filter_size in filter_sizes:
            filter_size = int(filter_size)
            one_channel = nn.ModuleList()
            tmp = nn.Conv1d(self.word_rep.feature_size, self.word_rep.feature_size, kernel_size=filter_size,
                            padding=int(floor(filter_size / 2)))
            xavier_uniform(tmp.weight)
            one_channel.add_module('baseconv', tmp)
            tmp = AttentionBolckV2(self.word_rep.feature_size, args.pool_size, True)
            one_channel.add_module('basevonb-pool', tmp)

            conv_dimension = self.word_rep.conv_dict[args.conv_layer]
            for idx in range(args.conv_layer):
                tmp = ResidualBlockHidden(conv_dimension[idx], conv_dimension[idx + 1], filter_size, 1, True, args.dropout if args.use_transformer else 0.0, use_layer_norm=args.use_layer_norm, is_relu=args.use_relu)
                one_channel.add_module('resconv-{}'.format(idx), tmp)
            self.conv.add_module('channel-{}'.format(filter_size), one_channel)
        if args.output_layer == 'att':
            self.output_layer = OutputLayerMulti(args, Y, dicts, self.filter_num * args.num_filter_maps)
        elif args.output_layer == 'attre':
            self.output_layer = OutputLayerMultiRe(args, Y, dicts, self.filter_num * args.num_filter_maps)
        elif args.output_layer == 'prob':
            self.output_layer = OutputLayerPro(args, Y, dicts, self.filter_num * args.num_filter_maps)
        elif args.output_layer == 'probtg':
            self.output_layer = OutputLayerProTg(args, Y, dicts, self.filter_num * args.num_filter_maps)
        elif args.output_layer == 'base':
            self.output_layer = OutputLayerMultiBase(args, Y, dicts, self.filter_num * args.num_filter_maps)
        else:
            self.output_layer = OutputLayerMultiNaive(args, Y, dicts, self.filter_num * args.num_filter_maps)
        if args.use_transformer:
            self.transfer = EncoderHidden(len(dicts['e2ind']), args.MAX_LENGTH, args.transfer_layer, self.filter_num * args.num_filter_maps, args.transfer_attention_head, args.transfer_fsize, args.dropout, gpu=args.gpu)
#
    def forward(self, x, target_lv0, target_lv1, target_lv2, text_inputs):

        x = self.word_rep(x, None, text_inputs, self.use_elmo)

        x = x.transpose(1, 2)

        conv_result = []
        for conv in self.conv:
            tmp = x
            for idx, md in enumerate(conv):
                if idx == 0:
                    tmp = torch.tanh(md(tmp))
                else:
                    tmp = md(tmp)
            tmp = tmp.transpose(1, 2)
            conv_result.append(tmp)
        x = torch.cat(conv_result, dim=2)
        if self.use_transformer:
            x = self.transfer(x)
        y1, y2, y3, loss = self.output_layer(x, target_lv0, target_lv1, target_lv2, None)

        return y1, y2, y3, loss

    def freeze_net(self):
        for p in self.word_rep.embed.parameters():
            p.requires_grad = False

import os
class Bert_seq_cls(nn.Module):

    def __init__(self, args, Y):
        super(Bert_seq_cls, self).__init__()

        print("loading pretrained bert from {}".format(args.bert_dir))
        config_file = os.path.join(args.bert_dir, 'bert_config.json')
        self.config = BertConfig.from_json_file(config_file)
        print("Model config {}".format(self.config))
        self.bert = BertModel.from_pretrained(args.bert_dir)

        self.dim_reduction = nn.Linear(self.config.hidden_size, args.num_filter_maps)
        self.dropout = nn.Dropout(self.config.hidden_dropout_prob)
        self.classifier = nn.Linear(args.num_filter_maps, Y)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids, attention_mask, target):
        _, pooled_output = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        x = self.dim_reduction(pooled_output)
        x = self.dropout(x)
        y = self.classifier(x)

        loss = F.binary_cross_entropy_with_logits(y, target)
        return y, loss

    def init_bert_weights(self, module):

        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, BertLayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def freeze_net(self):
        pass


def pick_model(args, dicts):
    Y = len(dicts['ind2c'])
    if args.model == 'CNN':
        model = CNN(args, Y, dicts)
    elif args.model == 'MultiCNN':
        model = MultiCNN(args, Y, dicts)
    elif args.model == 'ResCNN':
        model = ResCNN(args, Y, dicts)
    elif args.model == 'MultiResCNN':
        model = MultiResCNN(args, Y, dicts)
    elif args.model == 'bert_seq_cls':
        model = Bert_seq_cls(args, Y)
    elif args.model == 'EntityEH':
        model = MultiResCNNEH(args, Y, dicts)
    elif args.model == 'EntityFlow':
        model = MultiResCNNEntity(args, Y, dicts)
    elif args.model == 'EntityFlowHidden':
        model = MultiResCNNEntityHidden(args, Y, dicts)
    elif args.model == 'FlowHidden':
        model = MultiResCNNHidden(args, Y, dicts)
    elif args.model == 'FlowHiddenProbI':
        model = MultiResCNNHiddenProbI(args, Y, dicts)
    elif args.model == 'Hybrid':
        model = MultiResCNNHiddenHybrid(args, Y, dicts)
    else:
        raise RuntimeError("wrong model name")

    if args.test_model:
        sd = torch.load(args.test_model)
        model.load_state_dict(sd)
    if args.previous_model:
        sd = torch.load(args.previous_model)
        model.load_state_dict(sd)
    if args.gpu >= 0:
        model.cuda(args.gpu)
    return model
