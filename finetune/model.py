import collections
import torch
import torch.nn as nn
import catalyst.dl.callbacks as callbacks
from catalyst.utils.factory import UtilsFactory
from catalyst.dl.runner import AbstractModelRunner
from catalyst.models.resnet_encoder import ResnetEncoder
from catalyst.models.sequential import SequentialNet


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

class LossCallback(callbacks.Callback):
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


callbacks.__dict__["LossCallback"] = LossCallback


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
    def _batch_handler(*, dct, model):
        embeddings, logits = model(image=dct["image"])
        output = {"embeddings": embeddings, "logits": logits}
        return output
