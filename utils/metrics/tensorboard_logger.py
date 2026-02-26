'''
From Z. Zhuang et al.
https://github.com/ICEORY/PMF
'''

def tensorboard_logger(epoch,
                       mode,
                       recorder,
                       metrics_dict,
                       loss_dict,
                       lr,
                       mapped_cls_name):
    
    # Metrics
    mean_acc = metrics_dict['mean_acc']
    mean_recall = metrics_dict['mean_recall']
    mean_iou = metrics_dict['mean_iou']
    
    # Losses
    loss_meter_avg = loss_dict['loss_meter_avg']
    loss_focal = loss_dict['loss_focal']
    loss_lovasz = loss_dict['loss_lovasz']
    loss_boundary_weighted = loss_dict.get('loss_boundary_weighted', 0.0)
    loss_aux_weighted = loss_dict.get('loss_aux_weighted', 0.0)
    loss_component_sum_weighted = loss_dict.get('loss_component_sum_weighted', 0.0)

    def _as_float(x):
        return x.item() if hasattr(x, 'item') else float(x)

    loss_focal_v = _as_float(loss_focal)
    loss_lovasz_v = _as_float(loss_lovasz)
    loss_boundary_w_v = _as_float(loss_boundary_weighted)
    loss_aux_w_v = _as_float(loss_aux_weighted)
    loss_comp_sum_v = max(_as_float(loss_component_sum_weighted), 1e-12)

    recorder.tensorboard.add_scalar(
        tag='{}_Loss'.format(mode), scalar_value=loss_meter_avg, global_step=epoch)
    recorder.tensorboard.add_scalar(
        tag='{}_LossSoftmax'.format(mode), scalar_value=loss_focal_v, global_step=epoch)
    recorder.tensorboard.add_scalar(
        tag='{}_LossLovasz'.format(mode), scalar_value=loss_lovasz_v, global_step=epoch)
    recorder.tensorboard.add_scalar(
        tag='{}_LossBoundaryWeighted'.format(mode), scalar_value=loss_boundary_w_v, global_step=epoch)
    recorder.tensorboard.add_scalar(
        tag='{}_LossAuxWeighted'.format(mode), scalar_value=loss_aux_w_v, global_step=epoch)
    recorder.tensorboard.add_scalar(
        tag='{}_LossShareFocal'.format(mode), scalar_value=loss_focal_v / loss_comp_sum_v, global_step=epoch)
    recorder.tensorboard.add_scalar(
        tag='{}_LossShareLovasz'.format(mode), scalar_value=loss_lovasz_v / loss_comp_sum_v, global_step=epoch)
    recorder.tensorboard.add_scalar(
        tag='{}_LossShareBoundaryWeighted'.format(mode), scalar_value=loss_boundary_w_v / loss_comp_sum_v, global_step=epoch)
    recorder.tensorboard.add_scalar(
        tag='{}_LossShareAuxWeighted'.format(mode), scalar_value=loss_aux_w_v / loss_comp_sum_v, global_step=epoch)
    
    recorder.tensorboard.add_scalar(
        tag='{}_meanAcc'.format(mode), scalar_value=mean_acc.item(), global_step=epoch)
    recorder.tensorboard.add_scalar(
        tag='{}_meanIOU'.format(mode), scalar_value=mean_iou.item(), global_step=epoch)
    recorder.tensorboard.add_scalar(
        tag='{}_meanRecall'.format(mode), scalar_value=mean_recall.item(), global_step=epoch)
    recorder.tensorboard.add_scalar(
        tag='{}_lr'.format(mode), scalar_value=lr, global_step=epoch)
