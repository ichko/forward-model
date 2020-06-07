import os
import wandb
from src.utils import IS_DEBUG

if IS_DEBUG:
    os.environ['WANDB_MODE'] = 'dryrun'


class WAndBLogger:
    def __init__(self, info_log_interval, model, hparams, type):
        assert type in ['video', 'image'], \
            '`type` should be "video" or "image"'

        self.type = type
        self.info_log_interval = info_log_interval

        wandb.init(
            dir='.reports',
            project='forward_models_2',
            config=dict(
                hparams._get_kwargs(),
                name=model.name,
                model_num_params=model.count_parameters(),
            ),
        )

        # ScriptModule models can't be watched for the current version of wandb
        try:
            wandb.watch(model)
        except Exception as _e:
            pass

        # wandb.save('data.py')
        # wandb.save('utils.py')
        # wandb.save('run_experiment.py')
        # [
        #     wandb.save(f'./models/{f}') for f in os.listdir('./models')
        #     if not f.startswith('__')
        # ]
        self.model = model

    def log(self, dict):
        wandb.log(dict)

    def log_info(self, info, prefix='train'):
        if hasattr(self.model, 'scheduler'):
            wandb.log({'lr_scheduler': self.model.scheduler.get_lr()[0]})

        num_log_images = 2
        y = info['y'][:num_log_images].detach().cpu() * 255
        y_pred = info['y_pred'][:num_log_images].detach().cpu() * 255
        diff = abs(y - y_pred)

        wrapper_cls = wandb.Video if self.type == 'video' else wandb.Image

        wandb.log({
            f'{prefix}_y': [wrapper_cls(i) for i in y],
            f'{prefix}_y_pred': [wrapper_cls(i) for i in y_pred],
            f'{prefix}_diff': [wrapper_cls(i) for i in diff]
        })
