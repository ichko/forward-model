import wandb


class WAndBVideoLogger:
    def __init__(self, info_log_interval, model, hparams):
        self.info_log_interval = info_log_interval
        self.it = 0

        wandb.init(dir='.reports', project='forward_models_2', config=hparams)
        wandb.watch(model)
        # wandb.save('data.py')
        # wandb.save('utils.py')
        # wandb.save('run_experiment.py')
        # [
        #     wandb.save(f'./models/{f}') for f in os.listdir('./models')
        #     if not f.startswith('__')
        # ]

    def log_info(self, info):
        self.it += 1

        wandb.log({'loss': info['loss']})

        if self.it % self.info_log_interval == 0:
            num_log_images = 2
            y = info['y'][:num_log_images].detach().cpu()
            y_pred = info['y_pred'][:num_log_images].detach().cpu()
            diff = abs(y - y_pred)

            wandb.log({
                'y': [wandb.Video(i) for i in y],
                'y_pred': [wandb.Video(i) for i in y_pred],
                'diff': [wandb.Video(i) for i in diff]
            })
