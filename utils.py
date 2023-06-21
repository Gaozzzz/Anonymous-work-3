import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import auc, roc_curve, precision_recall_curve


def save_best_record(test_info, file_path):
    fo = open(file_path, "w")
    fo.write("Epoch: {}\n".format(test_info["epoch"][-1]))
    fo.write("auc: {:.4f}\n".format(test_info["Best_auc"][-1]))
    fo.write("ap: {:.4f}\n".format(test_info["Best_ap"][-1]))


class Loss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.bce = nn.BCELoss()

    def forward(self, result, _label):
        _label = _label.float()
        att = result['frame']
        anomaly = att.mean(-1)
        anomaly_loss = self.bce(anomaly, _label)
        cost = anomaly_loss
        return cost


def train(net, normal_loader, abnormal_loader, optimizer, criterion):
    net.train()
    net.flag = "Train"
    ninput, nlabel = next(normal_loader)
    ainput, alabel = next(abnormal_loader)
    _data = torch.cat((ninput, ainput), 0)
    _label = torch.cat((nlabel, alabel), 0)
    _data = _data.cuda()
    _label = _label.cuda()
    predict = net(_data)
    cost = criterion(predict, _label)
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()


def test(dataloader, model):
    with torch.no_grad():
        model.eval()
        model.flag = 'Test'
        pred = torch.zeros(0)
        gt = list(np.load('gt-colon.npy'))
        for i, (input, filename) in enumerate(dataloader):
            input = input.cuda()
            input = input.squeeze(2)
            pred_temp = torch.zeros(0)
            len_num_seg = input.shape[1]
            for j in range(input.shape[1] // 32 + 1):
                start_idx = j * 32
                end_idx = (j + 1) * 32
                input_tmp = input[:, start_idx:end_idx, :]
                if input_tmp.shape[1] < 32:
                    for last in range((32 - input_tmp.shape[1])):
                        input_tmp = torch.cat((input_tmp, input[:, -1, :].unsqueeze(1)), dim=1)
                predict = model(input_tmp)
                logits = torch.mean(predict['frame'], 0)
                sig = logits
                pred_temp = torch.cat((pred_temp, sig))
            pred = torch.cat((pred, pred_temp[:len_num_seg]))
        pred = list(pred.cpu().detach().numpy())
        pred = np.repeat(np.array(pred), 16)

        fpr, tpr, threshold = roc_curve(list(gt), pred)
        rec_auc = auc(fpr, tpr)
        precision, recall, th = precision_recall_curve(list(gt), pred)
        pr_auc = auc(recall, precision)

        return rec_auc, pr_auc


def val(dataloader, model):
    with torch.no_grad():
        model.eval()
        model.flag = 'Test'
        pred = torch.zeros(0)
        gt = list(np.load('gt-colon.npy'))
        predd = {}
        gtt = {}
        count = 0
        for i, (input, filename) in enumerate(dataloader):
            aa = filename[0].split('/')[-1]
            aa = aa.split('.')[0]
            aa = aa.split('_')[0]
            gtt[aa] = gt[count:count + len(input.mean(0).cpu().numpy()) * 16]
            count = count + input.shape[1] * 16
            input = input.cuda()
            input = input.squeeze(2)
            pred_temp = torch.zeros(0)
            len_num_seg = input.shape[1]
            for j in range(input.shape[1] // 32 + 1):
                start_idx = j * 32
                end_idx = (j + 1) * 32
                input_tmp = input[:, start_idx:end_idx, :]
                if input_tmp.shape[1] < 32:
                    for last in range((32 - input_tmp.shape[1])):
                        input_tmp = torch.cat((input_tmp, input[:, -1, :].unsqueeze(1)), dim=1)
                predict = model(input_tmp)
                logits = torch.mean(predict['frame'], 0)
                sig = logits
                pred_temp = torch.cat((pred_temp, sig))
            predd[aa] = np.repeat(pred_temp[:len_num_seg].cpu().detach().numpy(), 16)
            pred = torch.cat((pred, pred_temp[:len_num_seg]))

        pred = list(pred.cpu().detach().numpy())
        pred = np.repeat(np.array(pred), 16)

        fpr, tpr, threshold = roc_curve(list(gt), pred)
        rec_auc = auc(fpr, tpr)
        precision, recall, th = precision_recall_curve(list(gt), pred)
        pr_auc = auc(recall, precision)

        print(rec_auc, pr_auc)
