from .mmdet_train import custom_train_detector
# from mmseg.apis import train_segmentor
# from mmdet.apis import train_detector

from mmengine.logging import MMLogger
from torch.nn.parallel import DataParallel
from mmengine.model import MMDistributedDataParallel

from mmengine.runner import Runner
from mmengine.optim import build_optim_wrapper
from mmcv.transforms import Compose
from projects.mmdet3d_plugin.datasets.builder import build_dataloader

def custom_train_model(model,
                dataset,
                cfg,
                distributed=False,
                validate=False,
                timestamp=None,
                eval_model=None,
                meta=None):
    """A function wrapper for launching model training according to cfg.

    Because we need different eval_hook in runner. Should be deprecated in the
    future.
    """
    if cfg.model.type in ['EncoderDecoder3D']:
        assert False
    else:
        custom_train_detector(
            model,
            dataset,
            cfg,
            distributed=distributed,
            validate=validate,
            timestamp=timestamp,
            eval_model=eval_model,
            meta=meta)

def train_model(model,
                dataset,
                cfg,
                distributed=False,
                validate=False,
                timestamp=None,
                meta=None):
    """A function wrapper for launching model training according to cfg.

    Because we need different eval_hook in runner. Should be deprecated in the
    future.
    """
    logger = MMLogger.get_current_instance(cfg.log_level)

    # prepare data loaders
    dataset = dataset if isinstance(dataset, (list, tuple)) else [dataset]
    if 'imgs_per_gpu' in cfg.data:
        logger.warning('"imgs_per_gpu" is deprecated in MMDet V2.0. '
                       'Please use "samples_per_gpu" instead')
        if 'samples_per_gpu' in cfg.data:
            logger.warning(
                f'Got "imgs_per_gpu"={cfg.data.imgs_per_gpu} and '
                f'"samples_per_gpu"={cfg.data.samples_per_gpu}, "imgs_per_gpu"'
                f'={cfg.data.imgs_per_gpu} is used in this experiments')
        else:
            logger.warning(
                'Automatically set "samples_per_gpu"="imgs_per_gpu"='
                f'{cfg.data.imgs_per_gpu} in this experiments')
        cfg.data.samples_per_gpu = cfg.data.imgs_per_gpu

    data_loaders = [
        build_dataloader(
            ds,
            cfg.data.samples_per_gpu,
            cfg.data.workers_per_gpu,
            # cfg.gpus will be ignored if distributed
            len(cfg.gpu_ids),
            dist=distributed,
            seed=cfg.seed) for ds in dataset
    ]

    # put model on gpus
    if distributed:
        find_unused_parameters = cfg.get('find_unused_parameters', False)
        # Sets the `find_unused_parameters` parameter in
        # torch.nn.parallel.DistributedDataParallel
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False,
            find_unused_parameters=find_unused_parameters)
    else:
        model = DataParallel(
            model.cuda(cfg.gpu_ids[0]), device_ids=cfg.gpu_ids)

    # build runner
    optimizer = build_optim_wrapper(model, cfg.optimizer)

    cfg['model'] = model
    if eval_model is not None:
        cfg['eval_model'] = eval_model
    cfg['optim_wrapper'] = optimizer
    cfg['work_dir'] = cfg.work_dir
    cfg['logger'] = logger
    cfg['meta'] = meta
    cfg['default_hooks'] = dict(
        logger=dict(type='LoggerHook'),
        param_scheduler=dict(type='ParamSchedulerHook'),
        checkpoint=dict(type='CheckpointHook', interval=cfg.checkpoint_config.interval),
    )

    if distributed:
        if isinstance(runner, Runner):
            cfg['default_hooks']['sampler_seed'] = dict(type='DistSamplerSeedHook')

    if 'runner' not in cfg:
        cfg.runner = {
            'type': 'Runner',
            'max_epochs': cfg.total_epochs
        }
        warnings.warn(
            'config is now expected to have a `runner` section, '
            'please set `runner` in your config.', UserWarning)
    else:
        if 'total_epochs' in cfg:
            assert cfg.total_epochs == cfg.runner.max_epochs

    if cfg.resume_from:
        cfg['load_from'] = cfg.resume_from
        cfg['resume'] = True
    elif cfg.load_from:
        cfg['load_from'] = cfg.load_from
        cfg['resume'] = False
    
    # register eval hooks
    if validate:
        # Support batch_size > 1 in validation
        val_samples_per_gpu = cfg.data.val.pop('samples_per_gpu', 1)
        if val_samples_per_gpu > 1:
            # Replace 'ImageToTensor' to 'DefaultFormatBundle'
            cfg.data.val.pipeline = Compose(
                cfg.data.val.pipeline)
        val_dataset = build_dataset(cfg.data.val, dict(test_mode=True))
        val_dataloader = build_dataloader(
            val_dataset,
            samples_per_gpu=val_samples_per_gpu,
            workers_per_gpu=cfg.data.workers_per_gpu,
            dist=distributed,
            shuffle=False)

        cfg['val_dataloader'] = val_dataloader
        cfg['val_evaluator'] = dict(type='ToyAccuracyMetric')
        cfg['val_cfg'] = dict(type='ValLoop')

    if 'runner_type' not in cfg:
        # build the default runner
        runner = Runner.from_cfg(cfg)
    else:
        # build customized runner from the registry
        # if 'runner_type' is set in the cfg
        runner = RUNNERS.build(cfg)


    # an ugly workaround to make .log and .log.json filenames the same
    runner.timestamp = timestamp

    # # fp16 setting
    # TODO: enable fp16
    # fp16_cfg = cfg.get('fp16', None)
    # if fp16_cfg is not None:
    #     optimizer_config = Fp16OptimizerHook(
    #         **cfg.optimizer_config, **fp16_cfg, distributed=distributed)
    # elif distributed and 'type' not in cfg.optimizer_config:
    #     optimizer_config = OptimizerHook(**cfg.optimizer_config)
    # else:
    #     optimizer_config = cfg.optimizer_config



    # # user-defined hooks
    # TODO: enable custom hooks
    # if cfg.get('custom_hooks', None):
    #     custom_hooks = cfg.custom_hooks
    #     assert isinstance(custom_hooks, list), \
    #         f'custom_hooks expect list type, but got {type(custom_hooks)}'
    #     for hook_cfg in cfg.custom_hooks:
    #         assert isinstance(hook_cfg, dict), \
    #             'Each item in custom_hooks expects dict type, but got ' \
    #             f'{type(hook_cfg)}'
    #         hook_cfg = hook_cfg.copy()
    #         priority = hook_cfg.pop('priority', 'NORMAL')
    #         hook = build_from_cfg(hook_cfg, HOOKS)
    #         runner.register_hook(hook, priority=priority)

    # runner.run(data_loaders, cfg.workflow)
    runner.train()
