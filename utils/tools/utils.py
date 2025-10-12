import os
import torch
import torch.distributed as dist
import builtins
import datetime
import atexit


def setup_for_distributed(is_master):
    '''
    This function disables printing when not in master process
    '''
    builtin_print = builtins.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        force = force or (get_world_size() > 8)
        if is_master or force:
            now = datetime.datetime.now().time()
            builtin_print('[{}] '.format(now), end='')  # print with time stamp
            builtin_print(*args, **kwargs)

    builtins.print = print


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()

def is_main_process():
    return get_rank() == 0


def dist_barrier():
    if is_dist_avail_and_initialized():
        dist.barrier()


def init_distributed_mode(args):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ['RANK'])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
        args.distributed = True
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
        args.distributed = True
    else:
        print('Not using distributed mode')
        args.distributed = False
        args.rank = 0
        args.world_size = 1
        args.gpu = 0
        setup_for_distributed(is_master=True)
        return

    args.dist_backend = 'nccl'
    print('| distributed init (rank {}): {}'.format(args.rank, args.dist_url), flush=True)
    torch.distributed.init_process_group(
        backend=args.dist_backend,
        init_method=args.dist_url,
        world_size=args.world_size,
        rank=args.rank,
    )
    setup_for_distributed(args.rank == 0)
    # Ensure process group is properly torn down on exit
    try:
        atexit.register(cleanup)
    except Exception:
        pass

def cleanup():
    """
    Clean up distributed process group if initialized.
    Safe to call multiple times. Intended for use on normal exit and failures.
    """
    try:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()
    except Exception:
        # Swallow exceptions during cleanup to avoid masking originals
        pass

def setup_logger_for_distributed(is_master, logger):
    '''
    This function disables printing when not in master process
    '''

    logger_info = logger.info

    def info(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            logger_info(*args, **kwargs)

    logger.info = info

def setup_tensorboard_logger_for_distributed(is_master, tensorboard_logger):
    '''
    This function disables printing when not in master process
    '''

    tensorboard_logger_scalar_summary = tensorboard_logger.scalar_summary

    def scalar_summary(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            tensorboard_logger_scalar_summary(*args, **kwargs)

    tensorboard_logger.scalar_summary = scalar_summary
