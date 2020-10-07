import numpy as np
import random
import torch

from fastai.callback import Callback
from fastai.callbacks import *


class LogMetricsToAzure(Callback):
    def __init__(self, run, learner):
        self.run = run
        self.learner = learner

    def on_epoch_end(self, last_loss, last_metrics, **kwargs):
        self.run.log("train_loss", float(last_loss))
        self.run.log("val_loss", float(last_metrics[0]))
        self.run.log("acc", float(last_metrics[1]))

    def on_epoch_begin(self, **kwargs):
        self.run.log("lr", float(self.learner.opt.lr))
        self.run.log("wd", float(self.learner.opt.wd))
        self.run.log("mom", float(self.learner.opt.mom))
        self.run.log("beta", float(self.learner.opt.beta))


class CustomSaveModelCallback(Callback):
    def __init__(self, learner):
        self.learner = learner
        self.val_loss = None

    def on_epoch_end(self, epoch, last_metrics, **kwargs):
        if (self.val_loss is None) or (self.val_loss > last_metrics[0]):
            self.val_loss = last_metrics[0]
            self.learner.save("bestmodel")

            
def set_random_seeds(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    return