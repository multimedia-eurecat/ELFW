from __future__ import division
import numpy as np
import os, math, time
from torch.autograd import Variable
from torch.optim.lr_scheduler import MultiStepLR
from visualize import LinePlotter
from models import GCN, ResnetFCN, DeepLabV3
from elfw import *
from utils import *
from metrics import *

# Rafael Redondo, Jaume Gibert - Eurecat (c) 2019
# -------------------------------------------------------------------------
               
# -------------------------------------------------------------------------

def TrainVal(trainLoader,valLoader,e,r,d,lr_m,w,m,a,S,K,R,M,l=None):

    # -------------------------------------------------------------------------<
    # Hyper parameters

    max_epochs          = e
    lr                  = r
    lr_decay            = d
    lr_milestones       = lr_m
    weight_decay        = w
    momentum            = m
    data_augmen         = a
    synth_augmen_ratio  = S
    model_type          = M
    gcn_levels          = l

    hyper_str = model_type
    if model_type == "gcn":
        hyper_str += '-levels_' + str(gcn_levels)


    hyper_str += "-classes_"        + trainLoader.dataset.classes_code    + \
                 "-valset_"         + str(trainLoader.dataset.valset)     + \
                 "-lr_"             + str(lr)                             + \
                 "-lrdecay_"        + str(lr_decay)                       + \
                 "-lrmilestones"

    for ms in lr_milestones:
        hyper_str += "_" + str(ms)

    hyper_str += "-wdecay_"         + str(weight_decay)                   + \
                 "-momentum_"       + str(momentum)
                 
    if data_augmen:
        hyper_str += "-dataaugment"

    synth_aug_str = ''
    if synth_augmen_ratio > 0 and trainLoader.dataset.augmentation_folders_id:
        synth_aug_str = "-" + trainLoader.dataset.augmentation_folders_id + "_" + str(synth_augmen_ratio)
        hyper_str += synth_aug_str

    resume_str = ''
    if R:
        resume_str = "-resumed_" + os.path.split(R)[-1]
        hyper_str += resume_str

    # -------------------------------------------------------------------------

    print("Hyper parameters:\n" + \
                  "   model type............... \033[1m" + str(model_type)                  + "\033[0m\n"\
                  "   classes used............. \033[1m" + trainLoader.dataset.classes_code + "\033[0m\n"\
                  "   validation set .......... \033[1m" + str(trainLoader.dataset.valset)  + "\033[0m\n"\
                  "   max epochs............... \033[1m" + str(max_epochs)                  + "\033[0m\n"\
                  "   learning rate............ \033[1m" + str(lr)                          + "\033[0m\n"\
                  "   lr decay................. \033[1m" + str(lr_decay)                    + "\033[0m\n"\
                  "   lr milestones............ \033[1m" + str(lr_milestones)               + "\033[0m\n"\
                  "   weight_decay............. \033[1m" + str(weight_decay)                + "\033[0m\n"\
                  "   momentum................. \033[1m" + str(momentum)                    + "\033[0m\n"\
                  "   data augmentation........ \033[1m" + str(data_augmen)                 + "\033[0m")

    if synth_augmen_ratio > 0:
            print("   synthetic augmentation... \033[1m" + str(synth_augmen_ratio) + "\033[0m")
            print("   synthetic folders........ \033[1m" + trainLoader.dataset.augmentation_folders_id + "\033[0m")

    if model_type == "gcn":
        print(\
                  "   GCN levels............... \033[1m" + str(gcn_levels)         + "\033[0m\n")
    if R:
        print(\
                  "   Resumed from............. \033[1m" + str(R)                   + "\033[0m\n")

    # -------------------------------------------------------------------------
    # Checkpoints storage

    check_mkdir(K)
    checkpoints = os.path.join(K, hyper_str)

    if check_mkdir(checkpoints):
        for filename in os.listdir(checkpoints):
            if filename.endswith('.pth'):
                os.remove(os.path.join(checkpoints, filename))

    # -------------------------------------------------------------------------
    # Classes

    num_classes = trainLoader.dataset.num_classes
    label_names = trainLoader.dataset.label_names

    # -------------------------------------------------------------------------
    # Network Model

    if model_type == "fcn":
        model = torch.nn.DataParallel(ResnetFCN(num_classes))
    elif model_type == "gcn":
        model = torch.nn.DataParallel(GCN(num_classes,gcn_levels))
    elif model_type == "deeplab":
        model = torch.nn.DataParallel(DeepLabV3(num_classes))
    else:
        print('Model type not found.')
        exit(-1)

    if R:
        model.load_state_dict(torch.load(R))

    model.cuda()

    # -------------------------------------------------------------------------
    # Class weights: make sure weights are Float, otherwise the torch's loss will complain

    class_weights = torch.tensor(trainLoader.dataset.get_class_balance_weights()).type(torch.FloatTensor)

    # -------------------------------------------------------------------------
    # Optimization criterion

    criterion = torch.nn.CrossEntropyLoss(class_weights.cuda())
    optimizer = torch.optim.SGD(model.parameters(),
                                lr=lr,
                                momentum=momentum,
                                weight_decay=weight_decay)

    # -------------------------------------------------------------------------
    # Schedulers

    scheduler  = MultiStepLR(optimizer, milestones=lr_milestones, gamma=lr_decay)
    early_stop = EarlyStop(30, aim='maximum')

    # -------------------------------------------------------------------------
    # Visdom: custom your environment title

    visdom_environment = "ELFW-" + model_type + \
                         "-classes_%s" % trainLoader.dataset.classes_code + \
                         "-vs_" + str(trainLoader.dataset.valset) + \
                         synth_aug_str + resume_str

    plotter = LinePlotter(visdom_environment)

    # -------------------------------------------------------------------------

    for epoch in range(max_epochs):

        model.train()
        console     = AverageConsole('Train', len(trainLoader))
        train_loss  = AverageMeter()
        train_acc   = AverageMeter()

        for i, (images, labels) in enumerate(trainLoader):

            console.snap()

            images = Variable(images).cuda()
            labels = Variable(labels).cuda()
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss.update(loss.data.cpu())
            _, predicted = torch.max(outputs.data, 1)
            train_acc.update( 100 * (predicted == labels).sum().item() / np.prod(labels.size()) )

            console.updateprint(i)
           
        plotter.plot(epoch, train_loss.avg, 'Loss',             'train')
        plotter.plot(epoch, train_acc.avg,  'Global Accuracy',  'train')

        # ---------------------------------------------------------------------------------
        model.eval()
        console         = AverageConsole('Eval', len(valLoader))
        val_loss        = AverageMeter()
        val_acc         = AverageMeter()
        TP, TN, FP, FN  = ZerosTFPN(num_classes)

        with torch.no_grad():

            for i, (images, labels) in enumerate(valLoader):

                console.snap()

                images = Variable(images).cuda()
                labels = Variable(labels).cuda()
                outputs = model(images)
                loss = criterion(outputs, labels)
                # No backward, No optimization
                val_loss.update(loss.data.cpu())
                _, predictions = torch.max(outputs.data, 1)
                val_acc.update( 100 * (predictions == labels).sum().item() / np.prod(labels.size()) )

                tp, tn, fp, fn = TrueFalsePositiveNegatives(labels, predictions, num_classes)
                TP += tp
                TN += tn
                FP += fp
                FN += fn
                console.updateprint(i)

        # Extended metrics
        val_pixel_acc               = PixelAccuracy(TP, FN)
        val_mean_acc, val_class_acc = MeanAccuracy(TP, FN)
        val_mean_iu, val_class_iu   = MeanIU(TP, FN, FP)
        val_freq_iu                 = FrequencyWeightedIU(TP, FN, FP)
        val_mean_f1, val_class_f1   = MeanF1Score(TP, FN, FP)

        plotter.plot(epoch, optimizer.param_groups[0]['lr'], 'Learning Rate', 'Learning Rate')
        plotter.plot(epoch, val_loss.avg,   'Loss',             'validation')
        plotter.plot(epoch, val_acc.avg,    'Global Accuracy',  'validation')
        plotter.plot(epoch, val_pixel_acc,  'Pixel Accuracy',   'validation')
        plotter.plot(epoch, val_mean_acc,   'Mean Accuracy',    'validation')
        plotter.plot(epoch, val_mean_iu,    'Mean IU',          'validation')
        plotter.plot(epoch, val_freq_iu,    'Freq Weighted IU', 'validation')
        plotter.plot(epoch, val_mean_f1,    'Mean F1Score',     'validation')

        for c in range(0,num_classes):
            plotter.plot(epoch, val_class_acc[c],   'Class Accuracy',   label_names[c])
            plotter.plot(epoch, val_class_iu[c],    'Class IU',         label_names[c])
            plotter.plot(epoch, val_class_f1[c],    'Class F1Score',    label_names[c])


        print("Epoch [\033[1m%d\033[0m] Loss: \033[1m%.5f\033[0m, Acc: \033[1m%.2f\033[0m" % (epoch, val_loss.avg, val_acc.avg))

        # ---------------------------------------------------------------------------------
        # LR update

        scheduler.step()

        # ---------------------------------------------------------------------------------
        # Saves checkpoints

        if not epoch % 10:
            checkpoint_name = os.path.join(checkpoints, model_type + "-epoch_" + str(epoch).zfill(4) + ".pth")
            torch.save(model.state_dict(), checkpoint_name)
            print("Saved checkpoint at " + checkpoint_name)

        # ---------------------------------------------------------------------------------
        # Exit conditions

        # Early stop
        if early_stop.step(val_mean_iu):    
            print("It's been a long time since we do not improve the training. Let's early stop it.")
            return

        # Divergence
        if math.isnan(train_loss.avg) or math.isnan(val_loss.avg):
            print("Loss is out of range o_0. Let's stop.")
            return

