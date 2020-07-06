import sys
import argparse
import pprint

import torch
from tqdm.auto import trange

from src.utils import IS_DEBUG
from src.loggers.wandb import WAndBLogger
from src.utils import get_example_rollout
from src.pipeline.config import get_hparams
from src.pipeline.common import get_model, get_data_generator


def train(config_id, hparams, from_main=False):
    pp = pprint.PrettyPrinter(4)

    print(f'## Start training with configuration "{hparams.model.upper()}"')
    pp.pprint(vars(hparams))

    if not IS_DEBUG and from_main:
        print('\n\nPress ENTER to continue')
        _ = input()
        print('...')

    train_data_generator = get_data_generator(
        env_name=hparams.env_name,
        bs=hparams.bs,
        min_seq_len=hparams.min_seq_len,
        max_seq_len=hparams.max_seq_len,
        frame_size=hparams.frame_size,
        moving_window_slices=hparams.moving_window_slices,
        num_processes=24,
    )
    val_data_generator = get_data_generator(
        env_name=hparams.env_name,
        bs=8,
        min_seq_len=min(64, hparams.min_seq_len),
        max_seq_len=min(64, hparams.max_seq_len),
        frame_size=hparams.frame_size,
        moving_window_slices=None,
        num_processes=4,
        stochasticity=0.2,
    )

    model = get_model(hparams)

    # if 'from_scratch' in sys.argv:
    #     try:
    #         model.preload_weights()
    #         print('>>> MODEL PRELOADED')
    #     except Exception as _e:
    #         print('>>> Could not preload! Starting from scratch.')

    model.configure_optim(lr=hparams.lr)
    model = model.to(hparams.device)


    logger = WAndBLogger(
        name=config_id,
        info_log_interval=hparams.log_interval,
        model=model,
        hparams=hparams,
        type='video',
    )

    print(model.summary())

    tr = trange(hparams.its)

    for it in tr:
        print(it)
        batch = next(train_data_generator)
        train_loss, train_info = model.optim_step(batch)
        train_loss = train_loss.item()

        progress_description = \
            f'Loss: {train_loss:.6f}, Buff size: {len(train_data_generator)}'
        tr.set_description(progress_description)

        logger.log({'train_loss': train_loss})

        val_batch = next(val_data_generator)

        if it % hparams.log_interval == 0:
            logger.log_info(train_info, prefix='train')

            with torch.no_grad():
                val_loss, val_info = model.optim_step(val_batch)

                logger.log({'val_loss': val_loss})
                logger.log_info(val_info, prefix='val')

                num_example_rollouts = 3
                logger.log_images('example_val_rollout', [
                    get_example_rollout(val_info, i)
                    for i in range(num_example_rollouts)
                ])

                model.persist()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config',
        help='id of configuration',
    )
    parser.add_argument('--from-scratch', action='store_false')
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    hparams = get_hparams(args.config)
    train(args.config, hparams, from_main=True)
