from ..swin_transformer_v2 import create_swin_v2


def create_swin_encoder(model_cfg):
    return create_swin_v2(model_cfg)
