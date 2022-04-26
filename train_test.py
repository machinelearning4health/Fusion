
import torch
import numpy as np
from utils import all_metrics, print_metrics, get_pcodes, precision_eval, adjust_prob, macro_accuracy_show
import pickle
from tqdm import tqdm
from constants import *
def train(args, model, optimizer, epoch, gpu, data_loader, dicts):

    print("EPOCH %d" % epoch)

    losses = []


    model.train()

    # loader
    data_iter = iter(data_loader)
    num_iter = len(data_loader)
    if args.fast_train:
        num_iter = num_iter//4
    for i in tqdm(range(num_iter)):

        if args.model.find("bert") != -1:

            inputs_id, inputs_entity, segments, masks, labels = next(data_iter)

            inputs_id, segments, masks, labels = torch.LongTensor(inputs_id), torch.LongTensor(segments), \
                                                 torch.LongTensor(masks), torch.FloatTensor(labels)

            if gpu >= 0:
                inputs_id, inputs_entity, segments, masks, labels = inputs_id.cuda(gpu), inputs_entity.cuda(gpu), segments.cuda(gpu), \
                                                     masks.cuda(gpu), labels.cuda(gpu)

            output, loss = model(inputs_id, segments, masks, labels)
        elif args.model.find("EntityEH") != -1 or args.model.find("EntityFlowHidden") != -1:

            inputs_id, inputs_entity, inputs_negation, labels, _, hids_ = next(data_iter)
            pcodes, pcodes_mask, p_answers = get_pcodes(hids_, dicts['c2ind'])
            inputs_id, labels = torch.LongTensor(inputs_id), torch.FloatTensor(labels)
            inputs_entity, inputs_negation = torch.LongTensor(inputs_entity), torch.LongTensor(inputs_negation)
            pcodes, pcodes_mask = torch.LongTensor(pcodes), torch.FloatTensor(pcodes_mask)
            if gpu >= 0:
                inputs_id, labels = inputs_id.cuda(gpu), labels.cuda(gpu)
                inputs_entity, inputs_negation = inputs_entity.cuda(gpu), inputs_negation.cuda(gpu)
                pcodes, pcodes_mask = pcodes.cuda(gpu), pcodes_mask.cuda(gpu)

            output, loss = model(inputs_id, inputs_entity, inputs_negation, labels, None, pcodes, pcodes_mask)
        elif args.model.find("EntityFlow") != -1:

            inputs_id, inputs_entity, inputs_negation, input_list_entity, input_list_negation, labels, _, hids_ = next(
                data_iter)
            pcodes, pcodes_mask, p_answers = get_pcodes(hids_, dicts['c2ind'])
            inputs_id, labels = torch.LongTensor(inputs_id), torch.FloatTensor(labels)
            inputs_entity, inputs_negation = torch.LongTensor(inputs_entity), torch.LongTensor(inputs_negation)
            input_list_entity, input_list_negation = torch.LongTensor(input_list_entity), torch.LongTensor(
                input_list_negation)
            pcodes, pcodes_mask = torch.LongTensor(pcodes), torch.FloatTensor(pcodes_mask)
            if gpu >= 0:
                inputs_id, labels = inputs_id.cuda(gpu), labels.cuda(gpu)
                inputs_entity, inputs_negation = inputs_entity.cuda(gpu), inputs_negation.cuda(gpu)
                input_list_entity, input_list_negation = input_list_entity.cuda(gpu), input_list_negation.cuda(gpu)
                pcodes, pcodes_mask = pcodes.cuda(gpu), pcodes_mask.cuda(gpu)

            output, loss = model(inputs_id, inputs_entity, inputs_negation, input_list_entity, input_list_negation,
                                 labels, None, pcodes, pcodes_mask)
        elif args.model.find("Hybrid") != -1:

            inputs_id, labels_lv0, labels_lv1, labels_lv2, text_inputs, hids_ = next(data_iter)

            inputs_id, labels_lv0, labels_lv1, labels_lv2, = torch.LongTensor(inputs_id), torch.FloatTensor(
                labels_lv0), torch.FloatTensor(labels_lv1), torch.FloatTensor(labels_lv2)

            if gpu >= 0:
                inputs_id, labels_lv0, labels_lv1, labels_lv2 = inputs_id.cuda(gpu), labels_lv0.cuda(
                    gpu), labels_lv1.cuda(gpu), labels_lv2.cuda(gpu)
                text_inputs['elmo_tokens'] = text_inputs['elmo_tokens'].cuda(gpu)

            output, _, _, loss = model(inputs_id, labels_lv0, labels_lv1, labels_lv2, text_inputs)
        else:
            inputs_id, labels, text_inputs, hids = next(data_iter)

            inputs_id, labels = torch.LongTensor(inputs_id), torch.FloatTensor(labels)

            if gpu >= 0:
                inputs_id, labels = inputs_id.cuda(gpu), labels.cuda(gpu)
                text_inputs['elmo_tokens'] = text_inputs['elmo_tokens'].cuda(gpu)

            output, loss = model(inputs_id, labels, text_inputs)

        optimizer.zero_grad()
        loss.backward()
        if i%500 == 0:
            print('Loss:%f' %(loss.item()))
        optimizer.step()

        losses.append(loss.item())

    return losses

def test_special(args, model, data_path, fold, gpu, dicts, data_loader):

    filename = data_path.replace('train', fold)
    print('file for evaluation: %s' % filename)
    num_labels = len(dicts['ind2c'])

    y, yhat, yhat_raw, hids, losses = [], [], [], [], []

    model.eval()

    # loader
    data_iter = iter(data_loader)
    num_iter = len(data_loader)
    for i in tqdm(range(num_iter)):
        with torch.no_grad():

            if args.model.find("bert") != -1:
                inputs_id, segments, masks, labels = next(data_iter)

                inputs_id, segments, masks, labels = torch.LongTensor(inputs_id), torch.LongTensor(segments), \
                                                     torch.LongTensor(masks), torch.FloatTensor(labels)

                if gpu >= 0:
                    inputs_id, segments, masks, labels = inputs_id.cuda(
                        gpu), segments.cuda(gpu), masks.cuda(gpu), labels.cuda(gpu)

                output, loss = model(inputs_id, segments, masks, labels)
            elif args.model.find("TimeFlow") != -1:

                inputs_id, inputs_entity, inputs_negation, labels, _, hids_ = next(data_iter)
                pcodes, pcodes_mask, p_answers = get_pcodes(hids_, dicts['c2ind'])
                inputs_id, labels = torch.LongTensor(inputs_id), torch.FloatTensor(labels)
                inputs_entity, inputs_negation = torch.LongTensor(inputs_entity), torch.LongTensor(inputs_negation)
                pcodes, pcodes_mask = torch.LongTensor(pcodes), torch.FloatTensor(pcodes_mask)
                if gpu >= 0:
                    inputs_id, labels = inputs_id.cuda(gpu), labels.cuda(gpu)
                    inputs_entity, inputs_negation = inputs_entity.cuda(gpu), inputs_negation.cuda(gpu)
                    pcodes, pcodes_mask = pcodes.cuda(gpu), pcodes_mask.cuda(gpu)

                output, loss = model(inputs_id, inputs_entity, inputs_negation, labels, None, pcodes, pcodes_mask)
            elif args.model.find("EntityFlow") != -1:

                inputs_id, inputs_entity, inputs_negation, input_list_entity, input_list_negation, labels, _, hids_ = next(data_iter)
                pcodes, pcodes_mask, p_answers = get_pcodes(hids_, dicts['c2ind'])
                inputs_id, labels = torch.LongTensor(inputs_id), torch.FloatTensor(labels)
                inputs_entity, inputs_negation = torch.LongTensor(inputs_entity), torch.LongTensor(inputs_negation)
                input_list_entity, input_list_negation = torch.LongTensor(input_list_entity), torch.LongTensor(input_list_negation)
                pcodes, pcodes_mask = torch.LongTensor(pcodes), torch.FloatTensor(pcodes_mask)
                if gpu >= 0:
                    inputs_id, labels = inputs_id.cuda(gpu), labels.cuda(gpu)
                    inputs_entity, inputs_negation = inputs_entity.cuda(gpu), inputs_negation.cuda(gpu)
                    input_list_entity, input_list_negation = input_list_entity.cuda(gpu), input_list_negation.cuda(gpu)
                    pcodes, pcodes_mask = pcodes.cuda(gpu), pcodes_mask.cuda(gpu)

                output, loss = model(inputs_id, inputs_entity, inputs_negation, input_list_entity, input_list_negation, labels, None, pcodes, pcodes_mask)
            elif args.model.find("Hybrid") != -1:

                inputs_id, labels_lv0, labels_lv1, labels_lv2, text_inputs, hids_ = next(data_iter)

                inputs_id, labels_lv0, labels_lv1, labels_lv2, = torch.LongTensor(inputs_id), torch.FloatTensor(labels_lv0), torch.FloatTensor(labels_lv1), torch.FloatTensor(labels_lv2)

                if gpu >= 0:
                    inputs_id, labels_lv0, labels_lv1, labels_lv2 = inputs_id.cuda(gpu), labels_lv0.cuda(gpu), labels_lv1.cuda(gpu), labels_lv2.cuda(gpu)

                output, loss = model(inputs_id, labels_lv0, labels_lv1, labels_lv2, text_inputs)
            else:

                inputs_id, labels, text_inputs, hids_ = next(data_iter)

                inputs_id, labels, = torch.LongTensor(inputs_id), torch.FloatTensor(labels)

                if gpu >= 0:
                    inputs_id, labels, text_inputs = inputs_id.cuda(gpu), labels.cuda(gpu), text_inputs.cuda(gpu)

                output, loss = model(inputs_id, labels, text_inputs)

            output = torch.sigmoid(output)
            output = output.data.cpu().numpy()

            losses.append(loss.item())
            target_data = labels.data.cpu().numpy()

            yhat_raw.append(output)
            output = np.round(output)
            y.append(target_data)
            yhat.append(output)
            hids.extend(hids_)

    y = np.concatenate(y, axis=0)
    yhat = np.concatenate(yhat, axis=0)
    yhat_raw = np.concatenate(yhat_raw, axis=0)
    hid2id = {}
    for index_id, hid in enumerate(hids):
        hid2id[hid] = index_id

    hids_threshold_list = pickle.load(open('%s/index_list_test.pkl' %MIMIC_3_DIR, 'rb'))
    index_list = []
    for hids_threshold in hids_threshold_list:
        temp_index = []
        for hid in hids_threshold:
            temp_index.append(hid2id[hid])
        index_list.append(temp_index)

    k = 5 if num_labels == 50 else [8,15]
    for index_c in index_list:
        yhat_c = yhat[index_c]
        y_c = y[index_c]
        yhat_raw_c = yhat_raw[index_c]
        metrics = all_metrics(yhat_c, y_c, k=k, yhat_raw=yhat_raw_c)
        print_metrics(metrics)
    return metrics

def test(args, model, data_path, fold, gpu, dicts, data_loader):

    filename = data_path.replace('train', fold)
    print('file for evaluation: %s' % filename)
    num_labels = len(dicts['ind2c'])

    y, yhat, yhat_raw, hids, losses, pre_icd, tar_icd, pre_icd_top10 = [], [], [], [], [], [], [], []
    y1, yhat1, yhat_raw1 = [], [], []
    y2, yhat2, yhat_raw2 = [], [], []
    model.eval()

    # loader
    data_iter = iter(data_loader)
    num_iter = len(data_loader)

    for i in tqdm(range(num_iter)):
        with torch.no_grad():

            if args.model.find("bert") != -1:
                inputs_id, segments, masks, labels = next(data_iter)

                inputs_id, segments, masks, labels = torch.LongTensor(inputs_id), torch.LongTensor(segments), \
                                                     torch.LongTensor(masks), torch.FloatTensor(labels)

                if gpu >= 0:
                    inputs_id, segments, masks, labels = inputs_id.cuda(
                        gpu), segments.cuda(gpu), masks.cuda(gpu), labels.cuda(gpu)

                output, loss = model(inputs_id, segments, masks, labels)

            elif args.model.find("EntityEH") != -1 or args.model.find("EntityFlowHidden") != -1:

                inputs_id, inputs_entity, inputs_negation, labels, _, hids_ = next(data_iter)
                pcodes, pcodes_mask, p_answers = get_pcodes(hids_, dicts['c2ind'])
                inputs_id, labels = torch.LongTensor(inputs_id), torch.FloatTensor(labels)
                inputs_entity, inputs_negation = torch.LongTensor(inputs_entity), torch.LongTensor(inputs_negation)
                pcodes, pcodes_mask = torch.LongTensor(pcodes), torch.FloatTensor(pcodes_mask)
                if gpu >= 0:
                    inputs_id, labels = inputs_id.cuda(gpu), labels.cuda(gpu)
                    inputs_entity, inputs_negation = inputs_entity.cuda(gpu), inputs_negation.cuda(gpu)
                    pcodes, pcodes_mask = pcodes.cuda(gpu), pcodes_mask.cuda(gpu)

                output, loss = model(inputs_id, inputs_entity, inputs_negation, labels, None, pcodes, pcodes_mask)
            elif args.model.find("EntityFlow") != -1:

                inputs_id, inputs_entity, inputs_negation, input_list_entity, input_list_negation, labels, _, hids_ = next(data_iter)
                pcodes, pcodes_mask, p_answers = get_pcodes(hids_, dicts['c2ind'])
                inputs_id, labels = torch.LongTensor(inputs_id), torch.FloatTensor(labels)
                inputs_entity, inputs_negation = torch.LongTensor(inputs_entity), torch.LongTensor(inputs_negation)
                input_list_entity, input_list_negation = torch.LongTensor(input_list_entity), torch.LongTensor(input_list_negation)
                pcodes, pcodes_mask = torch.LongTensor(pcodes), torch.FloatTensor(pcodes_mask)
                if gpu >= 0:
                    inputs_id, labels = inputs_id.cuda(gpu), labels.cuda(gpu)
                    inputs_entity, inputs_negation = inputs_entity.cuda(gpu), inputs_negation.cuda(gpu)
                    input_list_entity, input_list_negation = input_list_entity.cuda(gpu), input_list_negation.cuda(gpu)
                    pcodes, pcodes_mask = pcodes.cuda(gpu), pcodes_mask.cuda(gpu)

                output, loss = model(inputs_id, inputs_entity, inputs_negation, input_list_entity, input_list_negation, labels, None, pcodes, pcodes_mask)
            elif args.model.find("Hybrid") != -1:

                inputs_id, labels_lv0, labels_lv1, labels_lv2, text_inputs, hids_ = next(data_iter)

                inputs_id, labels_lv0, labels_lv1, labels_lv2, = torch.LongTensor(inputs_id), torch.FloatTensor(labels_lv0), torch.FloatTensor(labels_lv1), torch.FloatTensor(labels_lv2)

                if gpu >= 0:
                    inputs_id, labels_lv0, labels_lv1, labels_lv2 = inputs_id.cuda(gpu), labels_lv0.cuda(gpu), labels_lv1.cuda(gpu), labels_lv2.cuda(gpu)
                    text_inputs['elmo_tokens'] = text_inputs['elmo_tokens'].cuda(gpu)
                labels = labels_lv2
                output1, output2, output, loss = model(inputs_id, labels_lv0, labels_lv1, labels_lv2, text_inputs)

                output1 = torch.sigmoid(output1)
                output1 = output1.data.cpu().numpy()
                output2 = torch.sigmoid(output2)
                output2 = output2.data.cpu().numpy()

                target_data_lv1 = labels_lv0.data.cpu().numpy()
                target_data_lv2 = labels_lv1.data.cpu().numpy()

                yhat_raw1.append(output1)
                yhat_raw2.append(output2)
                output1 = np.round(output1)
                output2 = np.round(output2)

                y1.append(target_data_lv1)
                yhat1.append(output1)
                y2.append(target_data_lv2)
                yhat2.append(output2)

            else:

                inputs_id, labels, text_inputs, hids = next(data_iter)

                inputs_id, labels, = torch.LongTensor(inputs_id), torch.FloatTensor(labels)

                if gpu >= 0:
                    inputs_id, labels, text_inputs = inputs_id.cuda(gpu), labels.cuda(gpu), None
                output, loss = model(inputs_id, labels, text_inputs)

            output = torch.sigmoid(output)
            output = output.data.cpu().numpy()
            #output = adjust_prob(output, dicts['prob_matrix'], dicts['confidence_matrix'], dicts['ind2c'])
            losses.append(loss.item())
            target_data = labels.data.cpu().numpy()

            yhat_raw.append(output)
            output = np.round(output)

            y.append(target_data)
            yhat.append(output)

    if args.model.find("Hybrid") != -1:
        y1 = np.concatenate(y1, axis=0)
        yhat1 = np.concatenate(yhat1, axis=0)
        yhat_raw1 = np.concatenate(yhat_raw1, axis=0)
        k = 5 if num_labels == 50 else [8, 15]
        metrics = all_metrics(yhat1, y1, k=k, yhat_raw=yhat_raw1)
        print_metrics(metrics)

        y2 = np.concatenate(y2, axis=0)
        yhat2 = np.concatenate(yhat2, axis=0)
        yhat_raw2 = np.concatenate(yhat_raw2, axis=0)
        k = 5 if num_labels == 50 else [8, 15]
        metrics = all_metrics(yhat2, y2, k=k, yhat_raw=yhat_raw2)
        print_metrics(metrics)

    y = np.concatenate(y, axis=0)
    yhat = np.concatenate(yhat, axis=0)
    yhat_raw = np.concatenate(yhat_raw, axis=0)
    #precision_eval(yhat_raw, y, dicts['ind2c'])
    k = 5 if num_labels == 50 else [8, 15]
    metrics = all_metrics(yhat, y, k=k, yhat_raw=yhat_raw)
    macro_acc = macro_accuracy_show(yhat, y)
    print_metrics(metrics)
    return metrics