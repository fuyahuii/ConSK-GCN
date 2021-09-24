import copy
import time

import numpy as np
import torch
from tqdm import tqdm
from sklearn import metrics

import dgcn
import pandas as pd

log = dgcn.utils.get_logger()


def lr_scheduler(optimizer, epoch, lr_decay_epoch=60, decay_factor=0.1):
    """Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs."""
    if epoch % lr_decay_epoch == 0 and epoch > 1:
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * decay_factor
    return optimizer


class Coach:

    def __init__(self, trainset, testset, model, opt, args):
        self.trainset = trainset
        self.testset = testset
        self.model = model
        self.opt = opt
        self.args = args
        self.label_to_idx = {'hap': 2, 'sad': 3, 'neu': 0, 'ang': 1}
        self.best_test_f1 = None
        self.best_epoch = None
        self.best_state = None
        self.best_confusion_matrix = None

    def load_ckpt(self, ckpt):
        self.best_test_f1 = ckpt["best_test_f1"]
        self.best_epoch = ckpt["best_epoch"]
        self.best_state = ckpt["best_state"]
        self.best_confusion_matrix = ckpt["best_confusion_matrix"]
        self.model.load_state_dict(self.best_state)

    def train(self):
        log.debug(self.model)
        # Early stopping.
        best_test_f1, best_epoch, best_state, best_confusion_matrix = self.best_test_f1, self.best_epoch, self.best_state, self.best_confusion_matrix

        Loss_list = []
        Train_f1 = []
        Test_f1 = []
        self.optimizer=torch.optim.Adam(self.model.parameters(),lr=self.args.learning_rate, weight_decay=self.args.weight_decay)
        # Train
        for epoch in range(1, self.args.epochs + 1):
            loss, train_f1 = self.train_epoch(epoch)
            test_f1, test_confusion_matrix, golds, preds = self.test_evaluate(test=True)

            if best_test_f1 is None or test_f1 > best_test_f1:
                best_test_f1 = test_f1
                best_epoch = epoch
                best_confusion_matrix = test_confusion_matrix
                best_state = copy.deepcopy(self.model.state_dict())
                log.info("Save the best model.")

            self.optimizer = lr_scheduler(self.optimizer, epoch, 60, 0.1)
            test_f1, test_confusion_matrix, golds, preds = self.test_evaluate(test=True)
            log.info("[Test set] [f1 {:.4f}]".format(test_f1))

            Loss_list.append(loss)
            Train_f1.append(train_f1)
            Test_f1.append(test_f1)

        # The best
        self.model.load_state_dict(best_state)
        log.info("")
        log.info("Best in epoch {}:".format(best_epoch))
        # dev_f1 = self.evaluate()
        # log.info("[Dev set] [f1 {:.4f}]".format(dev_f1))
        test_f1, test_confusion_matrix, golds, preds = self.test_evaluate(test=True)
        gold_label = pd.DataFrame(data=golds)
        pred_label = pd.DataFrame(data=preds)
        gold_label.to_csv("gold_label.csv", mode='a+', index=None, header=None)
        pred_label.to_csv("pred_label_sem.csv", mode='a+', index=None, header=None)
        log.info("[Test set] [f1 {:.4f}]".format(test_f1))
        log.info("Test_confusion_matrix {}:".format(test_confusion_matrix))

        # x1=range(0,100)
        # y1=Loss_list
        # y2=Train_f1
        # y3=Test_f1
        # plt.subplot(3,1,1)
        # plt.plot(x1,y1,'o-')
        # plt.title('Loss vs. epoches')

        # plt.subplot(3, 1, 2)
        # plt.plot(x1, y2, '+--')
        # plt.title('Train accuracy vs. epoches')

        # plt.subplot(3, 1, 3)
        # plt.plot(x1, y3, '*:')
        # plt.title('Test accuracy vs. epoches')

        # plt.show()
        # plt.savefig("accuracy_loss.jpg")
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
                data[k] = v.to(self.args.device)
            nll = self.model.get_loss(data)
            epoch_loss += nll.item()
            nll.backward()
            self.opt.step()
        end_time = time.time()
        log.info("")
        log.info("[Epoch %d] [Loss: %f] [f1: %.4f] [Time: %f]" %
                 (epoch, epoch_loss, train_f1, end_time - start_time))
        return epoch_loss, train_f1

    def train_evaluate(self, train=True):
        dataset = self.trainset
        self.model.train()
        with torch.no_grad():
            golds = []
            preds = []
            # for idx in tqdm(range(len(dataset)), desc="test" ):
            for idx in tqdm(range(len(dataset)), desc="train"):
                # print("len(dataset)",len(dataset))
                data = dataset[idx]
                golds.append(data["label_tensor"])
                for k, v in data.items():
                    data[k] = v.to(self.args.device)
                y_hat = self.model(data)
                preds.append(y_hat.detach().to("cpu"))

            golds = torch.cat(golds, dim=-1).numpy()
            preds = torch.cat(preds, dim=-1).numpy()
            f1 = metrics.f1_score(golds, preds, average="weighted")
        return f1

    def test_evaluate(self, test=True):
        dataset = self.testset
        self.model.eval()
        with torch.no_grad():
            golds = []
            preds = []
            for idx in tqdm(range(len(dataset)), desc="test"):
                data = dataset[idx]
                golds.append(data["label_tensor"])
                for k, v in data.items():
                    data[k] = v.to(self.args.device)
                y_hat = self.model(data)
                preds.append(y_hat.detach().to("cpu"))

            golds = torch.cat(golds, dim=-1).numpy()
            preds = torch.cat(preds, dim=-1).numpy()
            f1 = metrics.f1_score(golds, preds, average="weighted")
            confusion_matrix = metrics.confusion_matrix(golds, preds)

        return f1, confusion_matrix, golds, preds
