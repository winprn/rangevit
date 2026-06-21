
python main.py config/kitti/ablation/robustness/config_robust_range_noise.yaml --checkpoint checkpoint/kitti_base/train/best_miou_model.pth --val_only
python main.py config/kitti/ablation/robustness/config_robust_point_dropout.yaml --checkpoint checkpoint/kitti_base/train/best_miou_model.pth --val_only
python main.py config/kitti/ablation/robustness/config_robust_beam_dropout.yaml --checkpoint checkpoint/kitti_base/train/best_miou_model.pth --val_only




python main.py config/kitti/ablation/robustness/config_robust_range_noise.yaml --checkpoint base_best_miou_model.pth  --val_only
python main.py config/kitti/ablation/robustness/config_robust_point_dropout.yaml --checkpoint base_best_miou_model.pth  --val_only
python main.py config/kitti/ablation/robustness/config_robust_beam_dropout.yaml --checkpoint base_best_miou_model.pth  --val_only