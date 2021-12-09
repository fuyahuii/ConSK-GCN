import copy
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm
from sklearn import metrics
import nni
import dgcn
import pandas as pd

log = dgcn.utils.get_logger()


class Coach:

    def __init__(self, trainset, devset, testset, model, opt, args):
        self.trainset = trainset
        self.devset = devset
        self.testset = testset
        self.model = model
        self.opt = opt
        self.args = args
        self.label_to_idx = {'neu': 0, 'ang': 1, 'dis': 2, 'joy': 3,'sur':4,'sad':5,'fear':6}
        self.best_test_f1 = None
        self.best_epoch = None
        self.best_state = None

    def load_ckpt(self, ckpt):
        self.best_test_f1 = ckpt["best_test_f1"]
        self.best_epoch = ckpt["best_epoch"]
        self.best_state = ckpt["best_state"]
        self.model.load_state_dict(self.best_state)

    def train(self):
        log.debug(self.model)
        # Early stopping.
        best_dev_f1, best_epoch, best_state = self.best_dev_f1, self.best_epoch, self.best_state

        Train_f1 = []
        Dev_f1 = []
        Test_f1 = []
        # Train
        for epoch in range(1, self.args["epochs"] + 1):
            loss, train_f1 = self.train_epoch(epoch)
            dev_f1,dev_confusion_matrix,dev_golds,dev_preds = self.evaluate()
            nni.report_intermediate_result(dev_f1)
            log.info("[Dev set] [f1 {:.4f}]".format(dev_f1))

            test_f1,test_confusion_matrix,golds,preds = self.evaluate(test=True)
            log.info("[Test set] [f1 {:.4f}]".format(test_f1))
            # report intermediate result
            nni.report_intermediate_result(test_f1)

            if best_dev_f1 is None or dev_f1 > best_dev_f1:
                best_dev_f1 = dev_f1
                best_epoch = epoch
                best_state = copy.deepcopy(self.model.state_dict())
                log.info("Save the best model.")

            Dev_f1.append(dev_f1)
            Train_f1.append(train_f1)
            Test_f1.append(test_f1)
            
        # The best
        self.model.load_state_dict(best_state)
        log.info("")
        log.info("Best in epoch {}:".format(best_epoch))
        dev_f1,dev_confusion_matrix,dev_golds,dev_preds = self.evaluate()
        log.info("[Dev set] [f1 {:.4f}]".format(dev_f1))
        test_f1,test_confusion_matrix,golds,preds = self.evaluate(test=True)
        gold_label = pd.DataFrame(data=golds)
        pred_label = pd.DataFrame(data=preds)
        gold_label.to_csv("gold_label.csv", mode='a+', index=None, header=None)
        pred_label.to_csv("pred_label.csv", mode='a+', index=None, header=None)
        log.info("[Test set] [f1 {:.4f}]".format(test_f1))
        log.info("Test_confusion_matrix {}:".format(test_confusion_matrix))

        nni.report_final_result(test_f1)
        log.debug('Final result is %g', test_f1)
        log.debug('Send final result done.')

        return best_test_f1, best_epoch, best_state

    def train_epoch(self, epoch):
        start_time = time.time()
        epoch_loss = 0
        self.model.train()
        train_f1 = self.train_evaluate(train=True)
        for idx in tqdm(range(len(self.trainset)), desc="train epoch {}".format(epoch)):
            self.model.zero_grad()
            data = self.trainset[idx]
            for k, v in data.items():
                data[k] = v.to(self.args["device"])
            nll = self.model.get_loss(data)
            epoch_loss += nll.item()
            nll.backward()
            self.opt.step()

        end_time = time.time()
        log.info("")
        log.info("[Epoch %d] [Loss: %f] [f1: %.4f] [Time: %f]" %
                 (epoch, epoch_loss, train_f1, end_time - start_time))
        return epoch_loss,train_f1

    def train_evaluate(self, train=True):
        dataset = self.trainset
        self.model.train()
        with torch.no_grad():
            golds = []
            preds = []
            for idx in tqdm(range(len(dataset)), desc="train"):
                data = dataset[idx]
                golds.append(data["label_tensor"])
                for k, v in data.items():
                    data[k] = v.to(self.args["device"])
                y_hat = self.model(data)
                preds.append(y_hat.detach().to("cpu"))

            golds = torch.cat(golds, dim=-1).numpy()
            preds = torch.cat(preds, dim=-1).numpy()
            f1 = metrics.f1_score(golds, preds, average="weighted")
        return f1

    def evaluate(self, test=False):
        dataset = self.testset if test else self.devset
        self.model.eval()
        with torch.no_grad():
            golds = []
            preds = []
            for idx in tqdm(range(len(dataset)), desc="test" if test else "dev"):
                data = dataset[idx]
                golds.append(data["label_tensor"])
                for k, v in data.items():
                    data[k] = v.to(self.args["device"])
                y_hat = self.model(data)
                preds.append(y_hat.detach().to("cpu"))

            golds = torch.cat(golds, dim=-1).numpy()
            preds = torch.cat(preds, dim=-1).numpy()
            f1 = metrics.f1_score(golds, preds, average="weighted")
            confusion_matrix = metrics.confusion_matrix(golds, preds)

        return f1,confusion_matrix,golds,preds

