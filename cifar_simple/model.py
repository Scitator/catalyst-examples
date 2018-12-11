import collections
import torch.nn as nn
import torch.nn.functional as F
from catalyst.utils.factory import UtilsFactory
from catalyst.dl.callbacks import (
    ClassificationLossCallback,
    Logger, TensorboardLogger,
    OptimizerCallback, SchedulerCallback, CheckpointCallback,
    PrecisionCallback, OneCycleLR)
from catalyst.dl.runner import ClassificationRunner


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def build_simple_model():
    net = Net()
    return net


NETWORKS = {
    "simple": build_simple_model
}


def prepare_model(config):
    return UtilsFactory.create_model(
        config=config, available_networks=NETWORKS)


class ModelRunner(ClassificationRunner):

    @staticmethod
    def prepare_callbacks(
            *, args, mode, stage=None,
            precision_args=None, reduce_metric=None, **kwargs):
        assert len(kwargs) == 0
        precision_args = precision_args or [1, 3, 5]

        callbacks = collections.OrderedDict()

        callbacks["loss"] = ClassificationLossCallback()
        callbacks["optimizer"] = OptimizerCallback()
        callbacks["precision"] = PrecisionCallback(
            precision_args=precision_args)

        # OneCylce custom scheduler callback
        callbacks["scheduler"] = OneCycleLR(
            cycle_len=args.epochs,
            div=3, cut_div=4, momentum_range=(0.95, 0.85))

        # Pytorch scheduler callback
        # callbacks["scheduler"] = SchedulerCallback(
        #     reduce_metric=reduce_metric)

        callbacks["saver"] = CheckpointCallback()
        callbacks["logger"] = Logger()
        callbacks["tflogger"] = TensorboardLogger()

        return callbacks
