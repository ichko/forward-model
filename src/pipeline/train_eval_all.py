import multiprocessing as mp
from datetime import datetime

from src.pipeline.config import configs, get_hparams
from src.pipeline.train import train
from src.pipeline.eval import evaluate
from src.utils import random_seed

results_file_name = f'.results/results-{datetime.now()}.txt'


def train_eval(config_id, hparams):
    random_seed()
    train(config_id, hparams)

    random_seed()
    mean, std = evaluate(hparams)

    with open(results_file_name, 'a+') as f:
        f.write(f'CONFIG: {config_id}\n')
        f.write(f'MSE: {mean:.6f} ± {std:.6f}\n')
        f.write(f'HPARAMS:{str(vars(hparams))}\n')
        f.write('========================\n')


def main():
    items = list(enumerate(configs.items()))
    for i, (config_id, _config) in items:

        print(f'Config {config_id} ({i + 1}/{len(items)})')

        hparams = get_hparams(config_id)
        experiment = mp.Process(target=train_eval, args=(
            config_id,
            hparams,
        ))

        experiment.start()
        experiment.join()

        print('========================')


if __name__ == '__main__':
    main()
