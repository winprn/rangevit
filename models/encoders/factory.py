from .vit import create_vit_encoder
from .swin import create_swin_encoder
from .tinyvim import TinyViMEncoder


def create_encoder(model_cfg, decoder_name=None):
    model_cfg = model_cfg.copy()
    backbone = model_cfg.get('backbone', 'vit_small_patch16_384')

    if backbone.startswith('swin'):
        return create_swin_encoder(model_cfg)

    if backbone.startswith('tinyvim'):
        # TinyViMAdapter expects backbone_name and handles capacity internally.
        model_cfg['backbone_name'] = backbone
        if decoder_name == 'fpn':
            model_cfg['use_fpn_decoder'] = True
        # Tell adapter whether we're loading pretrained stem weights.
        model_cfg['load_pretrained_stem'] = (model_cfg.get('pretrained_path') is not None)
        return TinyViMEncoder(**model_cfg)

    return create_vit_encoder(model_cfg)
