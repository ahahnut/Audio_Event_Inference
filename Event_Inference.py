import argparse
import os
import pickle
import sys
import time
import torch
import torch.nn as nn
import shutil
import numpy as np
import torchvision
import csv
import json
import torchaudio
import numpy as np
import scipy.signal
import torch
import torch.nn.functional
from torch.utils.data import Dataset
import random
import numpy as np
from scipy import stats as stats_func
from sklearn import metrics
import torch
import os
import datetime
import time
import torch
import numpy as np
import pickle
import argparse
import sys
import time
import torch
import shutil
import ast
import numpy as np
from torchvision.models.feature_extraction import create_feature_extractor

###===========================================================================================================
# Utility Functions
###-----------------------------------------------------------------------------------------------------------
# Read CSV Label Files
def make_index_dict(label_csv):
    index_lookup = {}
    with open(label_csv, 'r') as f:
        csv_reader = csv.DictReader(f)
        line_count = 0
        for row in csv_reader:
            index_lookup[row['mid']] = row['index']
            line_count += 1
    return index_lookup

###-----------------------------------------------------------------------------------------------------------
# mAUC calculation
def d_prime(auc):
    standard_normal = stats_func.norm()
    d_prime = standard_normal.ppf(auc) * np.sqrt(2.0)
    return d_prime

###-----------------------------------------------------------------------------------------------------------
# Comprehensive Performance Metrics Calculation (We need to reset performance metrics)
"""Calculate statistics including mAP, AUC, etc.

    Args:
      output: 2d array, (samples_num, classes_num)
      target: 2d array, (samples_num, classes_num)

    Returns:
      stats: list of statistic of each class.
"""
def calculate_stats(output, target):
    classes_num = target.shape[-1]
    stats = []
    # Class-wise statistics
    for k in range(classes_num):
        # Average precision
        avg_precision = metrics.average_precision_score(
            target[:, k], output[:, k], average=None)
        # AUC
        auc = metrics.roc_auc_score(target[:, k], output[:, k], average=None)
        # Accuracy
        acc = metrics.accuracy_score(np.argmax(target, 1), np.argmax(output, 1))
        # F1
        target_i = np.argmax(target, axis=1)
        output_i = np.argmax(output, axis=1)
        f1 = metrics.f1_score(target_i, output_i, average=None)
        # Precisions, recalls
        (precisions, recalls, thresholds) = metrics.precision_recall_curve(
            target[:, k], output[:, k])
        # FPR, TPR
        (fpr, tpr, thresholds) = metrics.roc_curve(target[:, k], output[:, k])

        save_every_steps = 1     # Sample statistics to reduce size
        dict = {'precisions': precisions[0::save_every_steps],
                'recalls': recalls[0::save_every_steps],
                'AP': avg_precision,
                'fpr': fpr[0::save_every_steps],
                'fnr': 1. - tpr[0::save_every_steps],
                'auc': auc,
                'acc': acc,
                'f1': f1
                }
        stats.append(dict)
    return stats


###===========================================================================================================
# Dataset Preparation (We need to change the label embedding scheme)
class VSDataset(Dataset):
    ###---------------------------------------------------------------------------------------------------
    # Data Load (Audio + Labels)
    def __init__(self, dataset_json_file, label_csv=None, audio_conf=None, raw_wav_mode=False, specaug=False):
        self.datapath = dataset_json_file
        with open(dataset_json_file, 'r') as fp:
            data_json = json.load(fp)
        self.data = data_json['data']
        self.audio_conf = audio_conf
        self.mode = self.audio_conf.get('mode')
        self.melbins = self.audio_conf.get('num_mel_bins')
        self.index_dict = make_index_dict(label_csv)
        self.label_num = len(self.index_dict)

        self.windows = {'hamming': scipy.signal.hamming, 'hann': scipy.signal.hann, 'blackman': scipy.signal.blackman, 'bartlett': scipy.signal.bartlett}

        # if just load raw wavform
        self.raw_wav_mode = raw_wav_mode
        if specaug == True:
            self.freqm = self.audio_conf.get('freqm')
            self.timem = self.audio_conf.get('timem')
            print('now using following mask: {:d} freq, {:d} time'.format(self.audio_conf.get('freqm'), self.audio_conf.get('timem')))
        self.specaug = specaug
        self.mixup = self.audio_conf.get('mixup')
    
    ###---------------------------------------------------------------------------------------------------
    # Transform Audio into fbank
    def _wav2fbank(self, filename, filename2=None):
        # not mix-up, the colab version remove the mixup part
        waveform, sr = torchaudio.load(filename)
        waveform = waveform - waveform.mean()

        fbank = torchaudio.compliance.kaldi.fbank(waveform, htk_compat=True, sample_frequency=sr, use_energy=False,
                                                  window_type='hanning', num_mel_bins=self.melbins, dither=0.0, frame_shift=10)

        target_length = self.audio_conf.get('target_length', 512)
        n_frames = fbank.shape[0]

        p = target_length - n_frames

        # cut and pad
        if p > 0:
            m = torch.nn.ZeroPad2d((0, 0, 0, p))
            fbank = m(fbank)
        elif p < 0:
            fbank = fbank[0:target_length, :]

        return fbank

    ###---------------------------------------------------------------------------------------------------
    # Prepare fbanks and Prepare one-hot labels
    def __getitem__(self, index):
        datum = self.data[index]
        label_indices = np.zeros(self.label_num) + 0.00
        fbank = self._wav2fbank(datum['wav'])
        for label_str in datum['labels'].split(','):
            label_indices[int(self.index_dict[label_str])] = 1.0
        label_indices = torch.FloatTensor(label_indices)

        if self.specaug == True:
            freqm = torchaudio.transforms.FrequencyMasking(self.freqm)
            timem = torchaudio.transforms.TimeMasking(self.timem)
            fbank = torch.transpose(fbank, 0, 1)
            fbank = fbank.unsqueeze(0)
            fbank = freqm(fbank)
            fbank = timem(fbank)
            fbank = fbank.squeeze(0)
            fbank = torch.transpose(fbank, 0, 1)

        # mean/std is get from the val set as a prior.
        fbank = (fbank + 3.05) / 5.42

        # shift if in the training set, training set typically use mixup
        if self.mode == 'train':
            fbank = torch.roll(fbank, np.random.randint(0, 1024), 0)

        return fbank, label_indices

    def __len__(self):
        return len(self.data)


###===========================================================================================================
# Model Preparation (We need to change all the used model structure)
class EffNetOri(torch.nn.Module):
    def __init__(self, label_dim=6, pretrain=True, level=0):
        super().__init__()
        b = int(level)
        if pretrain == True:
            print('now train a effnet-b{:d} model with ImageNet pretrain'.format(b))
        else:
            print('now train a effnet-b{:d} model without ImageNet pretrain'.format(b))
        if b == 7:
            self.model = torchvision.models.efficientnet_b7(pretrained=pretrain)
        elif b == 6:
            self.model = torchvision.models.efficientnet_b6(pretrained=pretrain)
        elif b == 5:
            self.model = torchvision.models.efficientnet_b5(pretrained=pretrain)
        elif b == 4:
            self.model = torchvision.models.efficientnet_b4(pretrained=pretrain)
        elif b == 3:
            self.model = torchvision.models.efficientnet_b3(pretrained=pretrain)
        elif b == 2:
            self.model = torchvision.models.efficientnet_b2(pretrained=pretrain)
        elif b == 1:
            self.model = torchvision.models.efficientnet_b1(pretrained=pretrain)
        elif b == 0:
            self.model = torchvision.models.efficientnet_b0(pretrained=pretrain)

        new_proj = torch.nn.Conv2d(3, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        print('conv1 get from pretrained model.')
        new_proj.weight = torch.nn.Parameter(torch.sum(self.model.features[0][0].weight, dim=1).unsqueeze(1))
        new_proj.bias = self.model.features[0][0].bias
        self.model.features[0][0] = new_proj
        self.model = create_feature_extractor(self.model, {'features.8': 'mout'})
        self.feat_dim, self.freq_dim = self.get_dim()
        self.linear = nn.Linear(self.feat_dim, label_dim)

    def get_dim(self):
        # expect input x = (batch_size, time_frame_num, frequency_bins), e.g., (12, 1024, 128)
        x = torch.zeros(10, 1000, 128)
        x = x.unsqueeze(1)
        x = x.transpose(2, 3)
        x = self.model(x)['mout']
        return int(x.shape[1]), int(x.shape[2])

    def forward(self, x):
        # expect input x = (batch_size, time_frame_num, frequency_bins), e.g., (12, 1024, 128)
        x = x.unsqueeze(1)
        x = x.transpose(2, 3)
        x = self.model(x)['mout']
        x = torch.mean(x, dim=[2, 3])
        x = self.linear(x)
        return x


###===========================================================================================================
# Train Function (We need to change the audio model loading and the performance metrics)
def train(audio_model, train_loader, test_loader, args):
    # Device Setting
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    # Measurement Initialize 
    best_epoch, best_cum_epoch, best_mAP, best_acc, best_cum_mAP = 0, 0, -np.inf, -np.inf, -np.inf
    global_step, epoch = 0, 0
    # Save Model Path
    exp_dir = args.exp_dir
    
    # Audio model setting and its parameters
    torch.set_grad_enabled(True)
    audio_model = audio_model.to(device)
    audio_trainables = [p for p in audio_model.parameters() if p.requires_grad]
    print('Total parameter number is : {:.3f} million'.format(sum(p.numel() for p in audio_model.parameters()) / 1000000))
    print('Total trainable parameter number is : {:.3f} million'.format(sum(p.numel() for p in audio_trainables) / 1000000))
    
    # Set up the optimizer and the corresponding hyperparameters
    trainables = audio_trainables
    optimizer = torch.optim.Adam(trainables, args.lr, weight_decay=args.weight_decay, betas=(0.95, 0.999))
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, list(range(10, 60)), gamma=1.0)
    
    ###-------------------------------------------------------------------------------------------------------
    # Train Recurrent Settings
    # result = np.zeros([args.n_epochs, 9])
    epoch += 1
    print("current #steps=%s, #epochs=%s" % (global_step, epoch))
    print("start training...")
    audio_model.train()
    while epoch < args.n_epochs + 1:
        audio_model.train()
        # Load training data using enumerate
        for i, (audio_input, labels) in enumerate(train_loader):
            # Move audio input and labels to device
            audio_input = audio_input.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            # Audio Model uses Audio Input to obtain the audio output/ And calculate the loss of audio model
            audio_output = audio_model(audio_input)
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(audio_output, torch.argmax(labels.long(), axis=1))
            # Optimization Initialization and Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            global_step += 1
            # end_time = time.time()
        
        # validation in one training recurrent step
        print('start validation')
        stats, valid_loss = validate(audio_model, test_loader, args, epoch)
        
        
        # cum_stats = stats
        # cum_mAP = np.mean([stat['AP'] for stat in cum_stats])
        # cum_mAUC = np.mean([stat['auc'] for stat in cum_stats])
        # cum_acc = np.mean([stat['acc'] for stat in cum_stats])
        
        # Calculate the Performance Metrics and Print the Metrics
        mAP = np.mean([stat['AP'] for stat in stats])
        mAUC = np.mean([stat['auc'] for stat in stats])
        acc = np.mean([stat['acc'] for stat in stats])
        middle_ps = [stat['precisions'][int(len(stat['precisions'])/2)] for stat in stats]
        middle_rs = [stat['recalls'][int(len(stat['recalls'])/2)] for stat in stats]
        average_precision = np.mean(middle_ps)
        average_recall = np.mean(middle_rs)
        print("---------------------Epoch {:d} Results---------------------".format(epoch))
        print("ACC: {:.6f}".format(acc))
        print("mAP: {:.6f}".format(mAP))
        print("AUC: {:.6f}".format(mAUC))
        print("Avg Precision: {:.6f}".format(average_precision))
        print("Avg Recall: {:.6f}".format(average_recall))
        print("d_prime: {:.6f}".format(d_prime(mAUC)))
        print("valid_loss: {:.6f}".format(valid_loss))
        
        # Save the best training model
        if acc > best_acc:
            best_acc = acc
            best_acc_epoch = epoch
            torch.save(audio_model.state_dict(), "%s/models/best_audio_model_36.pth" % (exp_dir))
        scheduler.step()
        epoch += 1
        
        # print('number of params groups:' + str(len(optimizer.param_groups)))
        # print('Epoch-{0} lr: {1}'.format(epoch, optimizer.param_groups[0]['lr']))
        # result[epoch-1, :] = [mAP, acc, average_precision, average_recall, d_prime(mAUC), valid_loss, cum_mAP, cum_acc, optimizer.param_groups[0]['lr']]


###===========================================================================================================
# Validate Function 
def validate(audio_model, val_loader, args, epoch):
    
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    audio_model = audio_model.to(device)
    audio_model.eval()

    A_predictions, A_targets, A_loss = [], [], []
    with torch.no_grad():
        for i, (audio_input, labels) in enumerate(val_loader):
            audio_input = audio_input.to(device)

            # compute output
            audio_output = audio_model(audio_input)
            predictions = audio_output.to('cpu').detach()

            A_predictions.append(predictions)
            A_targets.append(labels)

            # compute the loss
            labels = labels.to(device)
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(audio_output, torch.argmax(labels.long(), axis=1))
            A_loss.append(loss.to('cpu').detach())

        audio_output = torch.cat(A_predictions)
        target = torch.cat(A_targets)
        loss = np.mean(A_loss)
        stats = calculate_stats(audio_output, target)

    return stats, loss


###===========================================================================================================
print("I am process %s, running on %s: starting (%s)" % (
        os.getpid(), os.uname()[1], time.asctime()))
###===========================================================================================================
# I/O args
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--data-train", type=str, default='/home/hxu16/code/vocalsound/data/datafiles_36/tr.json', help="training data json")
parser.add_argument("--data-val", type=str, default='/home/hxu16/code/vocalsound/data/datafiles_36/val.json', help="validation data json")
parser.add_argument("--label-csv", type=str, default='/home/hxu16/code/vocalsound/data/class_labels_indices_vs_36.csv', help="csv with class labels")
parser.add_argument("--exp-dir", type=str, default="/home/hxu16/code/vocalsound/data/baseline_exp/", help="directory to dump experiments")
# training and optimization args
parser.add_argument("--optim", type=str, default="adam", help="training optimizer", choices=["sgd", "adam"])
parser.add_argument('-b', '--batch-size', default=80, type=int, metavar='N', help='mini-batch size (default: 100)')
parser.add_argument('-w', '--num-workers', default=2, type=int, metavar='NW', help='# of workers for dataloading (default: 8)')
parser.add_argument('--lr', '--learning-rate', default=1e-4, type=float, metavar='LR', help='initial learning rate')
parser.add_argument('--lr-decay', default=40, type=int, metavar='LRDECAY', help='Divide the learning rate by 10 every lr_decay epochs')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
parser.add_argument('--weight-decay', '--wd', default=5e-7, type=float, metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument("--n-epochs", type=int, default=20, help="number of maximum training epochs")
parser.add_argument("--n-print-steps", type=int, default=1, help="number of steps to print statistics")
# models args
parser.add_argument("--n_class", type=int, default=36, help="number of classes")
parser.add_argument('--save_model', help='save the models or not', type=ast.literal_eval, default='False')
parser.add_argument("--model", type=str, default='eff_mean', help="model")
parser.add_argument("--model_size", type=int, default=0, help="model size")
parser.add_argument('--imagenet_pretrain', help='if use pretrained imagenet efficient net', type=ast.literal_eval, default='False')
parser.add_argument('--freqm', help='frequency mask max length', type=int, default=48)
parser.add_argument('--timem', help='time mask max length', type=int, default=192)
parser.add_argument("--mixup", type=float, default=0, help="how many (0-1) samples need to be mixup during training")
args = parser.parse_args(args=[])

###===========================================================================================================
audio_conf = {'num_mel_bins': 128, 'target_length': 512, 'freqm': args.freqm, 'timem': args.timem, 'mixup': args.mixup, 'mode': 'train'}
train_loader = torch.utils.data.DataLoader(
    VSDataset(args.data_train, label_csv=args.label_csv, audio_conf=audio_conf, raw_wav_mode=False, specaug=True),
    batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)

val_audio_conf = {'num_mel_bins': 128, 'target_length': 512, 'mixup': 0, 'mode': 'test'}
val_loader = torch.utils.data.DataLoader(
    VSDataset(args.data_val, label_csv=args.label_csv, audio_conf=val_audio_conf, raw_wav_mode=False),
    batch_size=200, shuffle=False, num_workers=args.num_workers, pin_memory=True)

###===========================================================================================================
if args.model == 'eff_mean':
    audio_model = EffNetOri(label_dim=args.n_class, level=args.model_size, pretrain=args.imagenet_pretrain)
else:
    raise ValueError('Model Unrecognized')

###===========================================================================================================
# start training
if os.path.exists(args.exp_dir):
    print("Deleting existing experiment directory %s" % args.exp_dir)
    shutil.rmtree(args.exp_dir)
print("\nCreating experiment directory: %s" % args.exp_dir)
os.makedirs("%s/models" % args.exp_dir)
with open("%s/args.pkl" % args.exp_dir, "wb") as f:
    pickle.dump(args, f)
print('Now starting training for {:d} epochs'.format(args.n_epochs))
train(audio_model, train_loader, val_loader, args)

###===========================================================================================================
# model selected on the validation set
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
sd = torch.load(args.exp_dir + '/models/best_audio_model_36.pth', map_location=device)
audio_model.load_state_dict(sd)
all_res = []
# best model on the validation set, repeat to confirm
stats, _ = validate(audio_model, val_loader, args, 'valid_set')
# note it is NOT mean of class-wise accuracy
print('---------------evaluate on the validation set---------------')
for i in range(36):
    val_acc = stats[i]['acc']
    val_precision = stats[i]['AP']
    val_recall = np.mean(stats[i]['recalls'])
    val_f1 = np.mean(stats[i]['f1'])
    val_auc = stats[i]['auc']

    print('------------------------------')
    print("Accuracy: {:.6f}".format(val_acc))
    print("Precision: {:.6f}".format(val_precision))
    print("Recall: {:.6f}".format(val_recall))
    print("F1: {:.6f}".format(val_f1))
    print("Auc: {:.6f}".format(val_auc))
