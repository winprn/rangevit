1) config_kitti.yaml:
```yaml
# Model config
change this --> vit_backbone: 'vit_small_patch16_384'
change this --> image_size: [64, 384]  # random crop at train
change this --> window_size: [64, 384] # sliding window size
```
2) rangevit.py
make sure the model configuration in the rangevit.py is correct (line 268 - 288)
``` python
if backbone == 'vit_small_patch16_384':
    n_heads = 6
    n_layers = 12
    patch_size = 16
    dropout = 0.0
    drop_path_rate = 0.1
    d_model = 192 ## change this for example
elif backbone == 'vit_base_patch16_384':
    n_heads = 12
    n_layers = 12
    patch_size = 16
    dropout = 0.0
    drop_path_rate = 0.1
    d_model = 768
elif backbone == 'vit_large_patch16_384':
    n_heads = 16
    n_layers = 24
    patch_size = 16
    dropout = 0.0
    drop_path_rate = 0.1
    d_model = 1024
``` 