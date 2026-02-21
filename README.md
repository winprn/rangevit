# RangeViT

<div align='center'>

**RangeViT: Towards Vision Transformers for 3D Semantic Segmentation in Autonomous Driving** [[arXiv](https://arxiv.org/abs/2301.10222)] \
*Angelika Ando, Spyros Gidaris, Andrei Bursuc, Gilles Puy, Alexandre Boulch and Renaud Marlet* \
**CVPR 2023**

<div>
  <img width="100%" alt="RangeViT architecture" src="images/RangeViT_method.png">
</div>

</div>

## **Citing**

If you use our **RangeViT** code in your research, please consider citing:

```
@inproceedings{RangeViT,
  title={RangeViT: Towards Vision Transformers for 3D Semantic Segmentation in Autonomous Driving},
  author={Ando, Angelika and Gidaris, Spyros and Bursuc, Andrei and Puy, Gilles and Boulch, Alexandre and Marlet, Renaud},
  booktitle={CVPR},
  year={2023}
}
```

## **Results & Downloading pretrained RangeViT models**

Results of RangeViT on the nuScenes validation set and on the SemanticKITTI test set with different weight initializations.

In particular, we initialize RangeViT’s backbone with ViTs pretrained (a) on supervised ImageNet21k classification and fine-tuned on supervised image segmentation on Cityscapes with [Segmenter](https://github.com/rstrudel/segmenter) (entry Cityscapes) (b) on supervised [ImageNet21k](https://github.com/huggingface/pytorch-image-models) classification (entry IN21k), (c) with the [DINO](https://github.com/facebookresearch/dino) self-supervised approach on ImageNet1k (entry DINO), and (d) trained from scratch (entry Random). The Cityscapes pre-trained ViT encoder weights can be downloaded from [here](https://www.rocq.inria.fr/cluster-willow/rstrudel/segmenter/checkpoints/cityscapes/seg_small_linear/).

| Train data | Test data                 | Pre-trained weights    | mIoU (%)    | Download  | Config |
|------------| ----------------------- | ---------------------- | ------------| ----------|---------|
| nuScenes train set | nuScenes val set | Cityscapes             | 75.2        | [RangeViT model](https://github.com/valeoai/rangevit/releases/download/v1/model_nuscenes_cs_init.pth) | [config](https://github.com/valeoai/rangevit/blob/main/config_nusc.yaml) |
| nuScenes train set | nuScenes val set | IN21k                  | 74.8        | [RangeViT model](https://github.com/valeoai/rangevit/releases/download/v1/model_nuscenes_in21k_init.pth) | [config](https://github.com/valeoai/rangevit/blob/main/config_nusc.yaml) |
| nuScenes train set | nuScenes val set | DINO                   | 73.3        | [RangeViT model](https://github.com/valeoai/rangevit/releases/download/v1/model_nuscenes_dino_init.pth) | [config](https://github.com/valeoai/rangevit/blob/main/config_nusc.yaml) |
| nuScenes train set | nuScenes val set | Random                 | 72.4        | [RangeViT model](https://github.com/valeoai/rangevit/releases/download/v1/model_nuscenes_rand_init.pth) | [config](https://github.com/valeoai/rangevit/blob/main/config_nusc.yaml) |
| SemanticKITTI train+val set | SemanticKITTI test set  | Cityscapes             | 64.0        | [RangeViT model](https://github.com/valeoai/rangevit/releases/download/v1/model_skitti_trainval_cs_init_h256.pth) | [config](https://github.com/valeoai/rangevit/blob/main/config_kitti_trainval.yaml) |
| SemanticKITTI train set | SemanticKITTI val set  | Cityscapes              | 60.8        | [RangeViT model](https://github.com/valeoai/rangevit/releases/download/v1/model_skitti_train_cs_init_h128.pth) | [config](https://github.com/valeoai/rangevit/blob/main/config_kitti.yaml) |

Note that the positional embeddings are initialized with the corresponding pre-trained weights or randomly when training from scratch. The convolutional stem, the decoder and the 3D refiner layer are always randomly initialized.


## **Preparation**

Please install [PyTorch](https://pytorch.org/) and then install the [nuScenes devkit](https://github.com/nutonomy/nuscenes-devkit) with

```bash
pip install nuscenes-devkit
```    

Finally, install the requirements with
```bash
pip install -r requirements.txt
```  

## **Model Input and Stem**

The model consumes a range-view tensor with shape `B x C x H x W`.

1. Input channels (`C=5`) are built as:
`[range, x, y, z, intensity]`
Source: `dataset/range_view_loader.py`.

2. Input preprocessing:
The 5 channels are normalized with `sensor.img_mean` and `sensor.img_stds`, then multiplied by the valid projection mask (invalid pixels are zeroed).
Source: `dataset/range_view_loader.py`.

3. Spatial size:
`H, W` come from crop/full-image settings in config.
For example, `config_kitti_tinyvim.yaml` uses train crop `image_size: [64, 1024]` and full projection width `original_image_size: [64, 2048]`.

4. Forward signatures:
Non-KPConv path uses only the 2D input tensor: `model(input_feature)`.
KPConv path uses extra point-wise tensors: `model(input2d, px, py, points_xyz, knns, num_points)`.
Source: `train.py`, `models/rangevit_kpconv.py`.

5. Stem for TinyViM backbones:
When `vit_backbone` starts with `tinyvim`, the encoder is `TinyViMAdapter`, which uses TinyViM's own stem (not `ConvStem` from `models/stems.py`).
The TinyViM stem is:
`Conv2d_BN -> GELU -> Conv2d_BN -> GELU`
Source: `models/tinyvim/tinyvim.py`, `models/tinyvim_adapter.py`.

6. Current TinyViM stem downsampling:
Current adapter default is `stem_stride=(1, 1)`, so there is no stem downsample in height or width.
Then stage embedding transitions use `down_stride=(1, 2)`, so width is downsampled across stages while height is preserved.
Source: `models/tinyvim_adapter.py`, `models/tinyvim/tinyvim.py`.

## **Training**

To train on nuScenes or on SemanticKITTI, use (and modify if needed) the config file `config_nusc.yaml` or `config_kitti.yaml`, respectively. For instance, to train on nuScenes, run the following command: 

```bash
python -m torch.distributed.launch --nproc_per_node=4 --master_port=63545 \
    --use_env main.py 'config_nusc.yaml' \
    --data_root '<path_to_nuscenes_dataset>' \
    --save_path '<path_to_log>' \
    --pretrained_model '<path_to_image_pretrained_model.pth>'
```

The `--pretrained_model` argument specifies the image-pretrained ViT-encoder that is used for initializing the ViT-encoder of RangeViT. For instance, to use the ImageNet21k-pretrained ViT-S encoder set ``--pretrained_model "timmImageNet21k"``. For the other initialization cases, you will need to download the pretrained weights. Read the Results section above to see where to download these pretrained weights from. Note that for all ViT-encoder initialization cases the peak learning rate of RangeViT is ``0.0008``, apart from the DINO initialization, in which case the peak learning rate is ``0.0002``.

Similarly, to train on SemanticKITTI, run the following command:
```bash
python -m torch.distributed.launch --nproc_per_node=4 --master_port=63545 \
    --use_env main.py 'config_kitti.yaml' \
    --data_root '<path_to_nuscenes_dataset>/dataset/sequences/' \
    --save_path '<path_to_log>' \
    --pretrained_model '<path_to_image_pretrained_model.pth>'
```

## **MLflow Integration**

Set the `mlflow` section in the configuration file to enable experiment tracking. A minimal example is:

```yaml
mlflow:
  enable: true
  tracking_uri: 'file:./mlruns'
  experiment_name: 'RangeViT'
  run_name: 'my_run'
  log_checkpoints: true
  log_code_snapshot: true
```

When enabled, the training loop logs epoch metrics, configuration files, code snapshots and the best checkpoints to the configured MLflow tracking server.
Both the original YAML file (as provided via the CLI) and a snapshot of the parsed configuration are stored under the `config/` artifact directory for reference.

## **Evaluation**

The same config files can be used for evaluating the pre-trained RangeViT models. 
For instance, to evaluate on the nuScenes validation set, run the following command:

```bash
python -m torch.distributed.launch --nproc_per_node=1 --master_port=63545 \
    --use_env main.py 'config_nusc.yaml' \
    --data_root '<path_to_nuscenes_dataset>' \
    --save_path '<path_to_log>' \
    --checkpoint '<path_to_pretrained_rangevit_model.pth>' \
    --val_only
```

To evaluate on the SemanticKITTI validation set, run the following command (adding the ``--test_split`` and ``--save_eval_results`` arguements for evaluating on the test split and saving the prediction results):

```bash
python -m torch.distributed.launch --nproc_per_node=1 --master_port=63545 \
    --use_env main.py 'config_kitti.yaml' \
    --data_root '<path_to_semantic_kitti_dataset>' \
    --save_path '<path_to_log>' \
    --checkpoint '<path_to_pretrained_rangevit_model.pth>' \
    --val_only
```
