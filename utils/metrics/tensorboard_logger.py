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

    recorder.tensorboard.add_scalar(
        tag='{}_Loss'.format(mode), scalar_value=loss_meter_avg, global_step=epoch)
    recorder.tensorboard.add_scalar(
        tag='{}_LossSoftmax'.format(mode), scalar_value=loss_focal.item(), global_step=epoch)
    recorder.tensorboard.add_scalar(
        tag='{}_LossLovasz'.format(mode), scalar_value=loss_lovasz.item(), global_step=epoch)
    
    recorder.tensorboard.add_scalar(
        tag='{}_meanAcc'.format(mode), scalar_value=mean_acc.item(), global_step=epoch)
    recorder.tensorboard.add_scalar(
        tag='{}_meanIOU'.format(mode), scalar_value=mean_iou.item(), global_step=epoch)
    recorder.tensorboard.add_scalar(
        tag='{}_meanRecall'.format(mode), scalar_value=mean_recall.item(), global_step=epoch)
    recorder.tensorboard.add_scalar(
        tag='{}_lr'.format(mode), scalar_value=lr, global_step=epoch)
