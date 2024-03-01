import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
import os.path as osp

from utils.util import EarlyStopping, save_file, set_gpu_devices, pause, set_seed
from utils.logger import logger
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data.dataloader import default_collate
from networks.CBSR_net import MyModel
from transformers import BertTokenizer, BertConfig
import h5py
import numpy as np
import pandas as pd
import pickle as pk
from tqdm import tqdm

parser = argparse.ArgumentParser(description="wxz codes")
parser.add_argument('--dataset', default='nextqa', choices=["nextqa"], type=str)
parser.add_argument('--mode', default="train", choices=["train", "test"], type=str)
parser.add_argument('--app_feat', default='resnet', choices=['resnet'], type=str)
parser.add_argument('--mot_feat', default='resnext', choices=['resnext'], type=str)

parser.add_argument("--video_path", type=str, default='./datasets/nextqa/')
parser.add_argument("--obj_path", type=str, default='./datasets/nextqa/')
parser.add_argument("--save_dir", type=str, default='./checkpoint/')
parser.add_argument("--csv_path", type=str, default='./csv_path/')
parser.add_argument("--qk_path", type=str, default="./datasets/nextqa/")

parser.add_argument("--amax_words", type=int, default=8)
parser.add_argument("--mc", type=int, help="num of candidates", default=5)
parser.add_argument("--num_bbox", type=int, help="the num of object in each frame", default=20)
parser.add_argument("--weight_path", type=str, default="")
# parser.add_argument("--weight_path", type=str, default="/home/wxz/CBSR/myCodes/CBSR/checkpoint/best_CBSR_nextqa_mix_61.01.ckpt")

parser.add_argument("--lr", type=float, default=1e-5, help='1e-5')
parser.add_argument("--bsize", type=int, help="BATCH_SIZE", default=32)
parser.add_argument("--epoch", type=int, help="epoch for train", default=20)
parser.add_argument("--num_workers", type=int, help="load dataset num", default=2)
parser.add_argument("--h_dim", type=int, help="hidden dim of vq encoder", default=768)
parser.add_argument("--dpout", type=float, help="dropout rate", default=0.2)
parser.add_argument("--weight_decay", type=float, help="weight_decay", default=0.001)
parser.add_argument("--pa", type=int, help="patience of ReduceonPleatu", default=1)

parser.add_argument("--grad_clip", type=int, help="", default=1)
parser.add_argument("--alpha", type=int, help="object num", default=0.6)
parser.add_argument("--seed", type=int, default=999)

args = parser.parse_args()
set_seed(args.seed)


def tokenize(seq, tokenizer, add_special_tokens=True, max_length=10, dynamic_padding=True, truncation=True):
    """
    :param seq: sequence of sequences of text
    :param tokenizer: bert_tokenizer
    :return: torch tensor padded up to length max_length of bert tokens
    """
    tokens = tokenizer.batch_encode_plus(
        seq,
        add_special_tokens=add_special_tokens,
        max_length=max_length,
        padding="longest" if dynamic_padding else "max_length",
        truncation=truncation,
    )["input_ids"]
    return torch.tensor(tokens, dtype=torch.long)


class VideoQADataset(Dataset):
    def __init__(self, dataname, csv_path, qk_path, amax_words, bert_tokenizer=None, mc=5, video_path="", obj_path="", obj_num=20):
        self.data = pd.read_csv(csv_path)
        self.amax_words = amax_words
        self.bert_tokenizer = bert_tokenizer
        self.mc = mc
        self.mode = osp.basename(csv_path).split('.')[0]  # train, val or test

        know_path = osp.join(qk_path, '{}_{}_know_bert.pt'.format(dataname, self.mode))
        qus_path = osp.join(qk_path, 'nextqa_{}_qus_bert.pt'.format(self.mode))
        con_path = osp.join(qk_path, 'nextqa_{}_con_bert.pt'.format(self.mode))

        know_vid, know_content = [], []
        with open("/home/wxz/LLM/mPLUG_img/Knowleged/NextQA_knowledge_4_punctuation.txt", 'r') as fread:
            fcontent = fread.readlines()
        for cur_con in fcontent:
            tmp = cur_con.strip().split("==>>")
            know_vid.append(int(tmp[0]))
            know_content.append(tmp[1])
        self.dict_vid_know = {}
        for cur_id, cur_know in zip(know_vid, know_content):
            self.dict_vid_know[cur_id] = cur_know

        self.qid2qus = {}
        with open(qus_path, 'rb') as fpt:
            self.sample_list = pk.load(fpt)
            self.question_id = self.sample_list['q_id']
            self.question_len = self.sample_list['q_len']
            self.question_feat = self.sample_list['q_feat'][:]
            self.len_dataset = len(self.question_id)
            print('Load Question: {}, Shape:{}'.format(qus_path, self.question_feat.shape))
            for id, (qid, qfeat, qlen) in enumerate(zip(self.question_id, self.question_feat, self.question_len)):
                self.qid2qus[str(qid)] = (qfeat, qlen)

        self.vid2know = {}
        with open(know_path, 'rb') as fpt:
            self.know_list = pk.load(fpt)
            self.know_vid = self.know_list["v_id"]
            self.know_feat = self.know_list['k_feat']
            self.know_len = self.know_list["k_len"]
            print('Load Knowledge: {}, Shape: {}'.format(know_path, self.know_feat.shape))
            for id, (vid, feat, length) in enumerate(zip(self.know_vid, self.know_feat, self.know_len)):
                self.vid2know[str(vid)] = (feat, length)

        self.qid2con = {}
        with open(con_path, 'rb') as fpt:
            self.con_list = pk.load(fpt)
            self.con_qid = self.con_list["q_id"]
            self.con_feat = self.con_list["qk_feat"][:]
            self.con_len = self.con_list["qk_len"]
            print("Load Con_feat: {} Shape: {}".format(con_path, self.con_feat.shape))
            for qid, cfeat, clen in zip(self.con_qid, self.con_feat, self.con_len):
                self.qid2con[str(qid)] = (cfeat, clen)

        # Loading Object
        self.bbox_feats = {}
        bbox_feat_file = osp.join(obj_path, f'acregion_8c{obj_num}b_{self.mode}.h5')
        with h5py.File(bbox_feat_file, 'r') as fp:
            vids = fp['ids']
            feats = fp['feat']
            print('Load Region: {}... Shape:{}'.format(bbox_feat_file, feats.shape))
            bboxes = fp['bbox']
            for id, (vid, feat, bbox) in enumerate(zip(vids, feats, bboxes)):
                self.bbox_feats[str(vid)] = (feat[:, :, :obj_num, :], bbox[:, :, :obj_num, :])

        # Loading Video
        app_path = osp.join(video_path, '{}_app_resnet.h5'.format(self.mode))
        mot_path = osp.join(video_path, '{}_mot_resnext.h5'.format(self.mode))
        print('Load Appearance {}...'.format(app_path))
        self.app_feats, self.mot_feats = {}, {}
        with h5py.File(app_path, 'r') as fp:
            for vid, feat in zip(fp['ids'][:], fp['feats'][:]):
                self.app_feats[str(vid)] = feat    # 字典，视频ID:视频外观特征

        print('Load Motion {}...'.format(mot_path))
        with h5py.File(mot_path, 'r') as fp:
            for vid, feat in zip(fp['ids'][:], fp['feats'][:]):
                self.mot_feats[str(vid)] = feat   # 字典，视频ID:视频运动特征

        print('-'*60)

    def transform_bb(self, roi_bbox, width, height):
        dshape = list(roi_bbox.shape)  # [8, 4, 20, 4]
        tmp_bbox = roi_bbox.reshape([-1, 4])  # [8*4*20, 4]
        relative_bbox = tmp_bbox / np.asarray([width, height, width, height])
        relative_area = (tmp_bbox[:, 2] - tmp_bbox[:, 0] + 1) * \
                        (tmp_bbox[:, 3] - tmp_bbox[:, 1] + 1) / (width * height)
        relative_area = relative_area.reshape(-1, 1)
        bbox_feat = np.hstack((relative_bbox, relative_area))
        dshape[-1] += 1
        bbox_feat = bbox_feat.reshape(dshape)

        return bbox_feat

    def get_video_feature(self, video_id,  width, height):
        app_feat = torch.from_numpy(self.app_feats[str(video_id)]).type(torch.float32)
        mot_feat = torch.from_numpy(self.mot_feats[str(video_id)]).type(torch.float32)

        (roi_feat, roi_bbox) = self.bbox_feats[str(video_id)]

        bbox_feat = self.transform_bb(roi_bbox, width, height)
        bbox_feat = torch.from_numpy(bbox_feat).type(torch.float32)     # [8, 4, 20, 5]
        roi_feat = torch.from_numpy(roi_feat).type(torch.float32)       # [8, 4, 20, 2048]

        region_feat = torch.cat((roi_feat, bbox_feat), dim=-1)

        return region_feat, app_feat, mot_feat

    def get_qus_know_according_candi(self, video_id, qus_id):
        (know_feat, know_len) = self.vid2know[str(video_id)]
        (qus_feat, qus_len) = self.qid2qus[str(qus_id)]
        (con_feat, con_len) = self.qid2con[str(qus_id)]

        return know_feat, know_len, qus_feat, qus_len, con_feat, con_len

    def __len__(self):
        return len(self.data['video_id'])

    def __getitem__(self, index):
        cur_sample = self.data.loc[index]
        video_id = int(cur_sample["video_id"])
        qid = str(cur_sample['qid'])
        question_txt = cur_sample['question']
        ans_id = int(cur_sample["answer"])
        width = int(cur_sample["width"])
        height = int(cur_sample["height"])
        qus_type = str(cur_sample['type'])

        qus_id = str(video_id) + '_' + qid

        answer_txts = [self.data["a" + str(i)][index] for i in range(self.mc)]
        candi_tokens = tokenize(answer_txts, self.bert_tokenizer, add_special_tokens=True,
                                max_length=self.amax_words, dynamic_padding=True, truncation=True)


        know_txt = self.dict_vid_know[video_id]
        con_text = question_txt + know_txt

        con_token = tokenize([con_text], self.bert_tokenizer, add_special_tokens=True, max_length=40,
                                  dynamic_padding=True, truncation=True)

        candi_tokens = torch.cat((con_token.tile(5, 1), candi_tokens[:, 1:]), dim=-1)

        obj_feat, app_feat, mot_feat = self.get_video_feature(video_id, width, height)
        know_feat, know_len, qus_feat, qus_len, con_feat, con_len = self.get_qus_know_according_candi(video_id, qus_id)
        know_feat = torch.from_numpy(know_feat).type(torch.float32)
        qus_feat = torch.from_numpy(qus_feat).type(torch.float32)
        con_feat = torch.from_numpy(con_feat).type(torch.float32)

        candi_len = (candi_tokens > 0).sum(-1)

        return {
            "ans_id": ans_id,
            "qus_id": qus_id,
            "qus_type": qus_type,
            "qus_feat": qus_feat,
            "qus_len": qus_len,
            "candi_tokens": candi_tokens,
            "candi_len": candi_len,
            "obj_feat": obj_feat,
            "app_feat": app_feat,
            "mot_feat": mot_feat,
            "know_feat": know_feat,
            "know_len": know_len,
            "con_feat": con_feat,
            "con_len": con_len,
        }


def videoqa_collate_fn(batch):
    if not isinstance(batch[0]["candi_tokens"], int):
        amax_len = max(x["candi_tokens"].size(1) for x in batch)
        for i in range(len(batch)):
            if batch[i]["candi_tokens"].size(1) < amax_len:
                batch[i]["candi_tokens"] = torch.cat(
                    [
                        batch[i]["candi_tokens"],
                        torch.zeros(
                            (
                                batch[i]["candi_tokens"].size(0),
                                amax_len - batch[i]["candi_tokens"].size(1),
                            ),
                            dtype=torch.long,
                        ),
                    ],
                    -1,
                )

    return default_collate(batch)


def get_videoqa_loaders(args, bert_tokenizer):
    train_dataset = VideoQADataset(
        dataname=args.dataset,
        csv_path=osp.join(args.csv_path, "train.csv"),
        qk_path=args.qk_path,
        amax_words=args.amax_words,
        bert_tokenizer=bert_tokenizer,
        mc=args.mc,  # 5
        video_path=args.video_path,
        obj_path=args.obj_path,
        obj_num=args.num_bbox,
    )
    train_loader = DataLoader(dataset=train_dataset, batch_size=args.bsize, num_workers=args.num_workers,
                              shuffle=True, drop_last=False, pin_memory=False, collate_fn=videoqa_collate_fn)

    test_dataset = VideoQADataset(
        dataname=args.dataset,
        csv_path=osp.join(args.csv_path, 'test.csv'),
        qk_path=args.qk_path,
        amax_words=args.amax_words,
        bert_tokenizer=bert_tokenizer,
        mc=args.mc,
        video_path=args.video_path,
        obj_path=args.obj_path,
        obj_num=args.num_bbox,
    )
    test_loader = DataLoader(dataset=test_dataset, batch_size=args.bsize, num_workers=args.num_workers,
                             shuffle=False, drop_last=False, pin_memory=False, collate_fn=videoqa_collate_fn)

    return train_loader, test_loader


class CBSR:
    def __init__(self, args):
        self.bsize = args.bsize
        self.mode = args.mode
        self.grad_clip = args.grad_clip
        self.mc = args.mc
        bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        bert_config = BertConfig.from_pretrained("bert-base-uncased", output_hidden_states=True)

        self.model_CBSR = MyModel(args.h_dim, args.dpout, 2048, 128, args.alpha, bert_config)
        if args.weight_path == "":
            self.model_CBSR.cuda()
            # param_dicts = [
            #     {"params": [p for n, p in self.model_CBSR.named_parameters() if "adapter" not in n and p.requires_grad]},
            #     {"params": [p for n, p in self.model_CBSR.named_parameters() if "adapter" in n and p.requires_grad],
            #      "lr": args.lr_bert}]
            param_dicts = [p for n, p in self.model_CBSR.named_parameters() if p.requires_grad]
            self.opt_all = torch.optim.AdamW(params=param_dicts, lr=args.lr, weight_decay=args.weight_decay)

            self.criterion = nn.CrossEntropyLoss()
            self.MSE = nn.MSELoss(reduction="sum")
            self.criterion.cuda()
            self.MSE.cuda()
        else:
            self.model_CBSR.load_state_dict(torch.load(args.weight_path))
            for name, param in self.model_CBSR.named_parameters():
                param.requires_grad = False
            self.model_CBSR.cuda()

        self.train_loader, self.test_loader = get_videoqa_loaders(args, bert_tokenizer)

    def train_s(self):
        self.model_CBSR.train()
        epoch_loss = 0.0
        coll_Spred, coll_Sreverse, coll_Sa2a = 0.0, 0.0, 0.0
        prediction_list = []
        answer_list = []

        for iter, batch_i in enumerate(tqdm(self.train_loader)):
            app_batch, mot_batch, obj_feat = batch_i["app_feat"].cuda(), batch_i["mot_feat"].cuda(), batch_i["obj_feat"].cuda()
            ques_feat = batch_i["qus_feat"].cuda()
            know_feat = batch_i["know_feat"].cuda()
            con_feat = batch_i["con_feat"].cuda()
            candi_tokens, candi_len, ans_id = batch_i["candi_tokens"].cuda(), batch_i["candi_len"].cuda(), batch_i["ans_id"]

            bs, num_candi, seq_c = candi_tokens.shape

            self.opt_all.zero_grad()
            video_feat, candi_sent = self.model_CBSR(obj_feat, app_batch, mot_batch, candi_tokens,
                                                     ques_feat, know_feat, con_feat)

            idx_bs = torch.arange(bs)
            ans_label = torch.zeros(bs, 5)
            ans_label[idx_bs, ans_id] = 1.0
            pred_out = F.cosine_similarity(video_feat.unsqueeze(dim=1), candi_sent, dim=-1)  # [N, num_candi]
            loss_con = self.MSE(pred_out, ans_label.cuda())

            answer_gt = candi_sent[idx_bs, ans_id, :].unsqueeze(dim=2)      # [bs, dim, 1]
            mask = torch.ones((bs, num_candi, 768)).cuda()                  # [bs, 5, 768]
            mask[idx_bs, ans_id, :] = 0.0                                   # [bs, 5, 768]
            answer_neg = candi_sent * mask
            B = torch.zeros((bs, num_candi, 768)).cuda()
            B[idx_bs, ans_id, :] = video_feat.squeeze()
            answer_neg = answer_neg + B
            pred_reverse = F.cosine_similarity(answer_gt.transpose(1, 2), answer_neg, dim=-1)
            loss_reverse = self.MSE(pred_reverse, ans_label.cuda())

            loss_a2a = self.MSE(pred_out, pred_reverse)

            loss_all = loss_con + loss_reverse + loss_a2a
            loss_all.backward()
            nn.utils.clip_grad_norm_(self.model_CBSR.parameters(), max_norm=self.grad_clip)
            self.opt_all.step()

            epoch_loss += loss_all.item()
            coll_Spred += loss_con.item()
            coll_Sreverse += loss_reverse.item()
            coll_Sa2a += loss_a2a.item()

            prediction = torch.argmax(pred_out, dim=1)  # pred_1.view(self.args.bs, 5).max(-1)[1]
            prediction_list.append(prediction)
            answer_list.append(ans_id.cpu())

        predict_answers = torch.cat(prediction_list, dim=0).long().cpu()
        ref_answers = torch.cat(answer_list, dim=0).long()
        acc_num = torch.sum(predict_answers == ref_answers).numpy()

        return epoch_loss, coll_Spred, coll_Sreverse, coll_Sa2a, acc_num * 100.0/len(ref_answers)

    def eval_acc(self):
        self.model_CBSR.eval()
        prediction_list, answer_list, list_qid = [], [], []

        with torch.no_grad():
            for iter, batch_i in enumerate(tqdm(self.test_loader)):
                app_batch, mot_batch, obj_feat = batch_i["app_feat"].cuda(), batch_i["mot_feat"].cuda(), batch_i["obj_feat"].cuda()
                ques_feat = batch_i["qus_feat"].cuda()
                know_feat = batch_i["know_feat"].cuda()
                con_feat = batch_i["con_feat"].cuda()
                candi_tokens, candi_len, ans_id = batch_i["candi_tokens"].cuda(), batch_i["candi_len"].cuda(), batch_i["ans_id"]
                qus_id = batch_i["qus_id"]

                bs, num_candi, seq_c = candi_tokens.shape

                video_feat, candi_sent = self.model_CBSR(obj_feat, app_batch, mot_batch, candi_tokens,
                                                         ques_feat, know_feat, con_feat)

                pred_out = F.cosine_similarity(video_feat.unsqueeze(dim=1), candi_sent, dim=-1)  # [N, num_candi]
                prediction_list.append(torch.argmax(pred_out, dim=1))
                answer_list.append(ans_id.cpu())
                list_qid.extend(qus_id)

            predict_answers = torch.cat(prediction_list, dim=0).long().cpu()
            ref_answers = torch.cat(answer_list, dim=0).long()
            acc_num = torch.sum(predict_answers == ref_answers).numpy()

        return acc_num * 100.0 / len(ref_answers)


def main(config, sign):
    wxzM = CBSR(config)

    print("= = "*20)
    print("obj_used {}, frames {}".format(int(args.alpha*args.num_bbox), int(args.alpha*32)))
    print("Trainable params of CBSR {},  Bert_adatper {}".format(
        sum(p.numel() for p in wxzM.model_CBSR.parameters() if p.requires_grad),
        sum(p.numel() for p in wxzM.model_CBSR.bert_candi.parameters() if p.requires_grad))
    )
    print("= = "*20)

    if config.weight_path == "":
        # training
        scheduler = ReduceLROnPlateau(wxzM.opt_all, 'max', factor=0.5, patience=args.pa, verbose=True)
        test_acc, best_test_acc, best_epoch = 0.0, 0.0, 0
        for epoch in range(1, args.epoch+1):
            scheduler.step(test_acc)
            loss_tr, loss_Spred, loss_Sreverse, loss_Sa2a, acc_tr = wxzM.train_s()
            test_acc = wxzM.eval_acc()

            logger.debug("[{}/{}]=>[Lr={}][Tr: {:.3f} pred: {:.3f}; reverse:{:.3f}; a2a:{:.3f}] [Tr acc: {:.2f} [Test acc: {:.2f}]]".
                         format(epoch, args.epoch, wxzM.opt_all.param_groups[0]['lr'], loss_tr, loss_Spred, loss_Sreverse, loss_Sa2a, acc_tr, test_acc))
            if test_acc > best_test_acc:
                best_test_acc = test_acc
                best_epoch = epoch
                best_model_path = './checkpoint/best_CBSR_{}.ckpt'.format(args.dataset)
                torch.save(wxzM.model_CBSR.state_dict(), best_model_path)
            print('--'*70)
        print("Epoch: {}, best_test_acc:{}".format(best_epoch, best_test_acc))
    else:
        test_acc = wxzM.eval_acc()
        print("test_acc:{:.2f}".format(test_acc))


if __name__ == "__main__":
    logger, sign = logger(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('Device:', device)
    main(args, sign)