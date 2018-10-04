import collections
import numpy as np
import torch
import torch.nn as nn
from prometheus.utils.factory import UtilsFactory
from prometheus.dl.callbacks import (
    Callback, LoggerCallback, OptimizerCallback, InferCallback,
    CheckpointCallback, OneCycleLR, LRFinder, PrecisionCallback)
from prometheus.dl.runner import AbstractModelRunner
from prometheus.models.resnet_encoder import ResnetEncoder
from prometheus.models.sequential import SequentialNet
from prometheus.dl.state import RunnerState


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
    return f"{model_params['model']}" \
           f"-{model_params['img_encoder']['arch']}" \
           f"-{model_params['img_encoder']['pooling']}" \
           f"-{model_params['cls_net']['hiddens']}" \
           f"-{model_params['cls_net']['emb_size']}"


# ---- Callbacks ----

class StageCallback(Callback):
    def on_stage_init(self, model, stage):
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

        state.loss["main"] = loss


# ---- Runner ----

class ModelRunner(AbstractModelRunner):

    def _init_state(
            self, *,
            mode: str,
            stage: str = None,
            **kwargs) -> RunnerState:
        """
        Inner method for children's classes for state specific initialization.
        :return: RunnerState with all necessary parameters.
        """
        additional_kwargs = {}

        if mode == "train":
            additional_kwargs["criterion"] = self.criterion.get("main", None)

        return super()._init_state(mode=mode, stage=stage, **additional_kwargs)

    @staticmethod
    def prepare_callbacks(*, callbacks_params, args, mode, stage=None):
        callbacks = collections.OrderedDict()

        if mode == "train":
            if stage == "debug":
                callbacks["stage"] = StageCallback()
                callbacks["loss"] = LossCallback(
                    emb_l2_reg=callbacks_params.get("emb_l2_reg", -1))
                callbacks["optimizer"] = OptimizerCallback(
                    grad_clip=callbacks_params.get("grad_clip", None))
                callbacks["lr-finder"] = LRFinder(
                    final_lr=callbacks_params.get("final_lr", 0.1),
                    n_steps=callbacks_params.get("n_steps", None))
                callbacks["logger"] = LoggerCallback(
                    reset_step=callbacks_params.get("reset_step", False))
            else:
                callbacks["stage"] = StageCallback()
                callbacks["loss"] = LossCallback(
                    emb_l2_reg=callbacks_params.get("emb_l2_reg", -1))
                callbacks["optimizer"] = OptimizerCallback(
                    grad_clip=callbacks_params.get("grad_clip", None))
                callbacks["one-cycle"] = OneCycleLR(
                    cycle_len=args.epochs,
                    div=3, cut_div=4, momentum_range=(0.95, 0.85))
                callbacks["precision"] = PrecisionCallback(
                    precision_args=callbacks_params.get(
                        "precision_args", [1, 3, 5]))
                callbacks["logger"] = LoggerCallback(
                    reset_step=callbacks_params.get("reset_step", False))
                callbacks["saver"] = CheckpointCallback(
                    save_n_best=getattr(args, "save_n_best", 5),
                    resume=args.resume,
                    main_metric=callbacks_params.get("main_metric", "loss"),
                    minimize=callbacks_params.get("minimize_metric", True))
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
