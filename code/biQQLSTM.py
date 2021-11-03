#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 29 16:04:43 2020

@author: mgy
"""

# torch packages
import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.nn as nn
from preprocess import *
# python's own
import time
import numpy as np
import itertools
import pickle
import math
import gc
import faulthandler
import random
import sys
# scikit-learn
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import matthews_corrcoef
# I wrote the following
from model import save_model, load_model, get_model_setting, readGrid
from data import myDataset, readData
# matlib plot
import matplotlib.pyplot as plt
from pprint import pprint
from scipy import stats
GLOBAL_BEST_ACC = -math.inf

def softplus(x):
    return torch.sigmoid(torch.log2(1+torch.exp(x)))

def customized_loss(output, target, variance, counterfact, a=0.5, b=0.05):
    mseloss = 0.5 * torch.sum((output - target)**2)
    var = softplus(variance)
    nllloss = 0.5 * torch.sum( (torch.log2(var) + ((output - target)**2 / var)))
    gap = torch.sum(torch.abs(output - counterfact))
    return a * mseloss + (1 - a) * nllloss + b * gap


class Trainer(object):
    """Trainer."""
    def __init__(self, trainer_args, args_dict, gridSearch):
        self.n_epochs = args_dict["epochs"]
        self.batch_size = args_dict["batch_size"]
        self.validate = trainer_args["validate"]
        self.save_best_dev = trainer_args["save_best_dev"]
        self.use_cuda = trainer_args["use_cuda"]
        self.print_every_step = trainer_args["print_every_step"]
        self.optimizer = trainer_args["optimizer"]
        self.model_path = args_dict["model_path"]

        # set for saving the best grid search results
        self.grid = {**gridSearch, **args_dict}

        self.device = torch.device("cuda:0" if torch.cuda.is_available() and self.use_cuda else "cpu")

        self.training_size = args_dict["training_size"]
        self.valid_size = args_dict["valid_size"]

    def train(self, network, train_data, dev_data=None):
        # transfer model to gpu if available
        network = network.to(self.device)
        # if torch.cuda.device_count()>1:
        #     network = nn.DataParallel(network)

        train_loss, validate_loss = [], []
        # define Tester over dev data
        if self.validate:
            default_valid_args = {
                "uncertain":    self.grid["uncertain"],
                "batch_size":   self.batch_size,
                "use_cuda":     self.use_cuda,
                "valid_size":   self.valid_size,
                "output_size":  self.grid["output_size"],
                "lamda":        self.grid['lamda'],
                "beta":         self.grid['beta'],
                }
            validator = Tester(**default_valid_args)
        
        for epoch in range(1, self.n_epochs + 1):
            # turn on network training mode
            network.train()

            # one forward and backward pass
            epoch_train_loss = self._train_step(train_data, network, n_print=self.print_every_step, epoch=epoch)
            train_loss.append(epoch_train_loss)
            print('epoch {}:'.format(epoch), end=" ")

            # validation
            if self.validate:
                if dev_data is None:
                    raise RuntimeError(
                        "Self.validate is True in trainer, "
                        "but dev_data is None."
                        "Please provide the validation data.")
                test_acc, epoch_valid_loss  = validator.test(network, dev_data)
                validate_loss.append(epoch_valid_loss)
                if self.save_best_dev and self.best_eval_result(test_acc):
                    save_model(network, self.model_path, self.grid)
                    print("Saved better model selected by validation.")
            # gc.collect()

        self.plot_loss(train_loss, validate_loss)
        # del train_loss
        # del validate_loss

    def plot_loss(self, train_loss, valid_loss):
        
        fig = plt.figure()
        ax = plt.subplot(221)
        ax.set_title('train loss')
        ax.plot(train_loss,'r-')

        ax = plt.subplot(222)
        ax.set_title('validation loss') 
        ax.plot(valid_loss,'b.')

        plt.savefig('trainLoss_vs_validLoss_{}.pdf'.format(self.grid['learning_rate']))
        plt.close()


    def _train_step(self, data_iterator, network, **kwargs):
        """Training process in one epoch.
        """
        loss = 0
        loss_record = 0.0

        for i, data in enumerate(data_iterator):
            label = data[-1].to(self.device)
            X = data[0]
            embedX = perturb(X)

            # inputX = []
            # for x in embedX:
            #     for key,value in x.items():
            #         x[key] = value.to(self.device)
            #     inputX.append(x)

            inputX = [x.to(self.device) for x in embedX]

            self.optimizer.zero_grad()
            if self.grid["uncertain"]:
                returnValue, counterfacts = network(inputX)
                logit, var = returnValue[:,0].view(-1,1), returnValue[:,1].view(-1,1)
                counterfact = counterfacts[:,0].view(-1,1)
                loss = customized_loss(logit, label, var, counterfact, a=self.grid['lamda'], b=self.grid['beta'])
            else:
                logit, counterfacts = network(embedX)
                criterion = torch.nn.BCELoss()
                loss = criterion(logit, label)

            # print(loss)
            loss.backward()
            self.optimizer.step()

            loss_record += loss.cpu().item()

        return loss_record / len(data_iterator)

    def best_eval_result(self, test_acc):       
        """Check if the current epoch yields better validation results.

        :param test_loss, a floating number
        :return: bool, True means current results on dev set is the best.
        """
        global GLOBAL_BEST_ACC
        
        if test_acc > GLOBAL_BEST_ACC:
            GLOBAL_BEST_ACC = test_acc
            return True
        return False
        
class Tester(object):
    """Tester."""

    def __init__(self, **kwargs):
        self.batch_size = kwargs["batch_size"]
        self.use_cuda = kwargs["use_cuda"]
        self.testing_size = kwargs["valid_size"]
        self.device = torch.device("cuda:0" if torch.cuda.is_available() and self.use_cuda else "cpu")
        self.output_size = kwargs["output_size"]
        self.lamda = kwargs['lamda']
        self.beta = kwargs['beta']
        self.uncertain = kwargs["uncertain"]

    def test(self, network, dev_data):
        network = network.to(self.device)

        # turn on the testing mode; clean up the history
        network.eval()

        mean_acc, valid_loss = 0.0, 0.0
        pred_list, truth_list, var_list = [], [], []
        # predict
        for i, data in enumerate(dev_data):
            label = data[-1].to(self.device)
            X = data[0]
            embedX = perturb(X)
            embedX = [x.to(self.device) for x in embedX]
            with torch.no_grad():
                if self.uncertain:
                    returnValue, counterfacts = network(embedX)
                    logit, var = returnValue[:,0].view(-1,1), returnValue[:,1].view(-1,1)
                    counterfact = counterfacts[:,0].view(-1,1)
                    loss = customized_loss(logit, label, var, counterfact, a=self.lamda, b=self.beta)
                else:
                    logit, counterfacts = network(embedX)
                    criterion = torch.nn.BCELoss()
                    loss = criterion(logit, label)

                valid_loss += loss.cpu().item()

            _prediction = list(logit.cpu().numpy())
            pred_list.extend(_prediction)

            _truth = list(label.cpu().numpy())
            truth_list.extend(_truth)
            if self.uncertain:
                _var = list(var.cpu().numpy())
                var_list.extend(_var)

        acc, _, mcc = generate_acc(pred_list,truth_list)
        print("[Tester] Accuracy: {:.4f}, MCC: {:.4f}".format(acc, mcc))

        return acc, valid_loss / len(dev_data)


class Predictor(object):
    """An interface for predicting outputs based on trained models.
    """

    def __init__(self, testing_size, batch_size=128, use_cuda=True):
        self.batch_size = batch_size
        self.use_cuda = use_cuda
        self.testing_size = testing_size
        self.device = torch.device("cuda:0" if torch.cuda.is_available() and self.use_cuda else "cpu")
        self.grid = readGrid()
        # self.output_size = readGrid()["output_size"]

    def predict(self, network, test_data):
        network = network.to(self.device)

        # turn on the testing mode; clean up the history
        network.eval()

        truth_list, pred_list, var_list = [], [], []

        acc = 0.0

        for i, data in enumerate(test_data):
            label = data[-1].to(self.device)
            X = data[0]
            embedX = perturb(X)
            embedX = [x.to(self.device) for x in embedX]

            with torch.no_grad():
                if self.grid["uncertain"]:
                    returnValue, counterfacts = network(embedX)
                    logit, var = returnValue[:,0].view(-1,1), returnValue[:,1].view(-1,1)
                else:
                    logit, counterfacts = network(embedX)

            _prediction = list(logit.cpu().numpy())
            pred_list.extend(_prediction)

            _truth = list(label.cpu().numpy())
            truth_list.extend(_truth)
            if self.grid["uncertain"]:
                _var = list(var.cpu().numpy())
                var_list.extend(_var)

        return pred_list,truth_list,var_list

def sigmoid(x):
  return 1 / (1 + np.exp(-x))

def generate_acc(pred, truth):
    pred = [1 if item > .5 else 0 for item in pred]
    truth = [1 if item > .5 else 0 for item in truth]
    # print(len(pred), len(truth))
    # print(sum(pred), sum(truth))
    target_names = ['legit', 'hate']
    report = classification_report(truth, pred, target_names=target_names, digits=3)
    acc = accuracy_score(truth, pred)
    mcc = matthews_corrcoef(truth, pred)
    return acc, report, mcc

def train(args_dict, gridSearch, data):
    """Train model.
    """
    print("Training...")
    # load data
    data_train = data["data_train"]
    # define model
    args = {**args_dict, **gridSearch}
    model = get_model_setting(**args)
    # define trainer
    trainer_args = {
        "validate":         True,
        "save_best_dev":    True,
        "use_cuda":         True,
        "print_every_step": 1000,
        "optimizer": torch.optim.Adam(model.parameters(), lr=args['learning_rate'], weight_decay=args['l2_norm']),
    }
    trainer = Trainer(trainer_args, args_dict, gridSearch)
    
    # train
    data_val = data["data_valid"]
    trainer.train(model, data_train, dev_data=data_val)

    print('')

def infer(data):
    """Inference using model.
    """
    print("Predicting...")
    # define model
    with open('hyper.pkl','rb') as f:
        final_grid = pickle.load(f)
    model = get_model_setting(**final_grid)
    load_model(model, final_grid['model_path'])

    # define predictor
    predictor = Predictor(batch_size=final_grid["batch_size"], use_cuda=True,testing_size=data["testing_size"])

    # predict
    data_test = data["data_test"]
    y_pred,y_true, variance = predictor.predict(model, data_test)
    np.save("pred_prob.npy", y_pred)
    np.save("variance.npy", variance)

    y_pred = [1 if item > .5 else 0 for item in y_pred]
    acc, report, mcc = generate_acc(y_pred, y_true)
    print("[Final tester] Accuracy: {:.4f}, MCC: {:.4f}".format(acc, mcc))
    print(report)

    np.save('prediction.npy',y_pred)
    np.save('ground_truth.npy',y_true)

    return

def loadData(datatype:str, shuffle=False, batch_size=128, num_workers=8):
    # get dataset
    X, y = readData(datatype)
    data_set = myDataset(X,y)
    data_size = len(data_set.y)
    print("{} size: {}".format(datatype, data_size))
    data_loader = DataLoader(data_set, 
                            batch_size=batch_size, 
                            shuffle=shuffle, 
                            num_workers=num_workers
                            )
    return data_loader, data_size

def pre():
    """Pre-process model."""
    print("Pre-processing...")
    uncertain = True
    # things to be fixed
    args_dict = {
        "uncertain":    uncertain,
        "batch_size":   128,
        "epochs":       8,
        "inputDim":     768,
        "output_size":  1+uncertain,
        "model_path":   "model.pkl",
        }

    # things to be tuned
    gridSearch = {
        "lstm_out":         [512],
        "lstm_bidirection": [True],
        "lstm_layers":      [2],
        "window_size":      [2],
        "lstm_dropout":     [0.05],
        "linear_out1":      [512],
        "linear_out2":      [64],
        "dropout1":         [0.05],
        "dropout2":         [0.2],
        "learning_rate":    [3e-6], # 3e-6 
        "l2_norm":          [4e-6],
        "lamda":            [0.3], # 0.3
        "beta":             [0.05],
        }

    return args_dict, gridSearch

def search(args_dict, gridSearch, data):
    for grid in [dict(zip(gridSearch.keys(),v)) for v in itertools.product(*gridSearch.values())]:
        train(args_dict, grid, data)
        gc.collect()
    return True

def wrapUp(task:str, func, **args):
    start = time.time()
    returnObject = func(**args)
    end = time.time()
    period = end-start
    print("{}:".format(task), end="")
    print("It took {} hour {} min {} sec".format(period//3600,(period%3600)//60,int(period%60)))
    print("")
    gc.collect()
    return returnObject

def main():
    # setting up seeds
    seed = int(sys.argv[1])
    # manually fix random seq
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # preprocessing
    args_dict,gridSearch = wrapUp(task="Preprocessing", func=pre)
    
    # load training & validation
    data_train,training_size = wrapUp(task="load train", func=loadData, datatype="train", shuffle=True, batch_size=args_dict["batch_size"])
    data_valid,valid_size = wrapUp(task="load valid", func=loadData, datatype="valid", batch_size=args_dict["batch_size"])

    args_dict["training_size"] = training_size
    args_dict["valid_size"] = valid_size
    data = {"data_train":data_train, "data_valid":data_valid}
    
    # training
    Status = wrapUp(task="Training", func=search, args_dict=args_dict, gridSearch=gridSearch, data=data)

    # load testing
    data_test,testing_size = wrapUp(task="load test", func=loadData, datatype="test", batch_size=args_dict["batch_size"])
    data = {
        "data_test": data_test,
        "testing_size": testing_size,
        "model_path": args_dict["model_path"],
    }
    # testing
    Status = wrapUp(task="Predicting", func=infer, data=data)

    return True

# code starts here
if __name__ == "__main__":
    faulthandler.enable()
    main()
    faulthandler.disable()
    
