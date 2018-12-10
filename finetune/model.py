import collections
import torch
import torch.nn as nn
from catalyst.utils.factory import UtilsFactory
from catalyst.dl.callbacks import (
    Callback, InferCallback,
    BaseMetrics, Logger, TensorboardLogger,
    OptimizerCallback, SchedulerCallback, CheckpointCallback,
    PrecisionCallback, OneCycleLR, LRFinder)
from catalyst.dl.runner import AbstractModelRunner
from catalyst.models.resnet_encoder import ResnetEncoder
from catalyst.models.sequential import SequentialNet
from catalyst.dl.state import RunnerState


# ---- Model ----


class ClsNet(nn.Module):
    def __init__(
            self, enc, n_cls,
            hiddens, emb_size,
            activation_fn=torch.nn.ReLU,
            norm_fn=None, bias=True, dropout=None):
        super().__init__()
        self.encoder = enc
        self.emb_net = SequentialNet(
            hiddens=hiddens + [emb_size],
            activation_fn=activation_fn,
            norm_fn=norm_fn, bias=bias, dropout=dropout)
        self.head = nn.Linear(emb_size, n_cls, bias=True)

    def forward(self, *, image):
        features = self.encoder(image)
        embeddings = self.emb_net(features)
        logits = self.head(embeddings)
        return embeddings, logits


def build_baseline_model(img_encoder, cls_net):
    img_enc = ResnetEncoder(**img_encoder)
    net = ClsNet(enc=img_enc, **cls_net)
    return net


NETWORKS = {
    "baseline": build_baseline_model
}


def prepare_model(config):
    return UtilsFactory.create_model(
        config=config, available_networks=NETWORKS)


def prepare_logdir(config):
    model_params = config["model_params"]
    data_params = config["stages"]["data_params"]
    return f"{data_params['train_folds']}" \
           f"-{model_params['model']}" \
           f"-{model_params['img_encoder']['arch']}" \
           f"-{model_params['img_encoder']['pooling']}" \
           f"-{model_params['cls_net']['hiddens']}" \
           f"-{model_params['cls_net']['emb_size']}"


# ---- Callbacks ----

class LossCallback(Callback):
    def __init__(self, emb_l2_reg=-1):
        self.emb_l2_reg = emb_l2_reg

    def on_batch_end(self, state):
        embeddings = state.output["embeddings"]
        logits = state.output["logits"]

        loss = state.criterion(logits.float(), state.input["targets"].long())

        if self.emb_l2_reg > 0:
            loss += torch.mean(
                torch.norm(embeddings.float(), dim=1)) * self.emb_l2_reg

        state.loss = loss


# ---- Runner ----

class ModelRunner(AbstractModelRunner):

    @staticmethod
    def prepare_stage_model(*, model, stage, **kwargs):
        AbstractModelRunner.prepare_stage_model(
            model=model, stage=stage, **kwargs)
        model_ = model
        if isinstance(model, torch.nn.DataParallel):
            model_ = model_.module

        if stage in ["debug", "stage1"]:
            for param in model_.encoder.parameters():
                param.requires_grad = False
        elif stage == "stage2":
            for param in model_.encoder.parameters():
                param.requires_grad = True
        else:
            raise NotImplemented

    @staticmethod
    def prepare_callbacks(
            *, args, mode, stage=None,
            emb_l2_reg=-1,
            save_n_best=5,
            precision_args=None, reduce_metric=None,
            grad_clip=None,
            final_lr=0.1, n_steps=None, **kwargs):
        assert len(kwargs) == 0
        precision_args = precision_args or [1, 3, 5]

        callbacks = collections.OrderedDict()

        if mode == "train":
            if stage == "debug":
                callbacks["loss"] = LossCallback(emb_l2_reg=emb_l2_reg)
                callbacks["optimizer"] = OptimizerCallback(
                    grad_clip_params=grad_clip)
                callbacks["metrics"] = BaseMetrics()
                callbacks["lr-finder"] = LRFinder(
                    final_lr=final_lr,
                    n_steps=n_steps)
                callbacks["logger"] = Logger()
                callbacks["tflogger"] = TensorboardLogger()
            else:
                callbacks["loss"] = LossCallback(emb_l2_reg=emb_l2_reg)
                callbacks["optimizer"] = OptimizerCallback(
                    grad_clip_params=grad_clip)
                callbacks["metrics"] = BaseMetrics()
                callbacks["precision"] = PrecisionCallback(
                    precision_args=precision_args)

                # OneCylce custom scheduler callback
                callbacks["one-cycle"] = OneCycleLR(
                    cycle_len=args.epochs,
                    div=3, cut_div=4, momentum_range=(0.95, 0.85))

                # Pytorch scheduler callback
                # callbacks["scheduler"] = SchedulerCallback(
                #     reduce_metric=reduce_metric)

                callbacks["saver"] = CheckpointCallback(
                    save_n_best=save_n_best,
                    resume=args.resume)
                callbacks["logger"] = Logger()
                callbacks["tflogger"] = TensorboardLogger()
        elif mode == "infer":
            callbacks["saver"] = CheckpointCallback(resume=args.resume)
            callbacks["infer"] = InferCallback(out_prefix=args.out_prefix)
        else:
            raise NotImplementedError

        return callbacks

    @staticmethod
    def _batch_handler(*, dct, model):
        embeddings, logits = model(image=dct["image"])
        output = {"embeddings": embeddings, "logits": logits}
        return output
