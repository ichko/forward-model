def get_model(hparams):
    import importlib
    import gym
    import sneks

    model_module = importlib.import_module(f'src.models.{hparams.model}')
    model_module.sanity_check()

    env = gym.make(hparams.env_name)

    model = model_module.make_model(
        precondition_size=hparams.precondition_size,
        frame_size=hparams.frame_size,
        num_actions=env.action_space.n,
    )
    model.make_persisted(f'.models/{model.name}_{hparams.env_name}.h5')

    return model


def get_data_generator(hparams, num_processes=8):
    import random
    from src.data.mp_rollout_generator import preprocessed_mp_generator
    from src.data.pong import PONGAgent

    def pong_agent_ctor(env):
        return PONGAgent(env, stochasticity=random.uniform(0.8, 1))

    agent_ctor = None  # random agenet
    if 'TwoPlayerPong' in hparams.env_name:
        agent_ctor = pong_agent_ctor

    return preprocessed_mp_generator(
        env_name=hparams.env_name,
        bs=hparams.bs,
        min_seq_len=hparams.min_seq_len,
        max_seq_len=hparams.max_seq_len,
        agent_ctor=agent_ctor,
        frame_size=hparams.frame_size,
        num_processes=num_processes,
        buffer_size=512,
        moving_window_slices=hparams.moving_window_slices,
    )


def main(hparams):
    import sys
    from src.loggers.wandb import WAndBLogger

    import torch
    from tqdm.auto import trange

    from src.utils import get_example_rollout

    train_data_generator = get_data_generator(hparams, num_processes=10)
    val_data_generator = get_data_generator(hparams, num_processes=4)

    model = get_model(hparams)

    if '--from-scratch' not in sys.argv:
        try:
            model.preload_weights()
            print('>>> MODEL PRELOADED')
        except Exception as _e:
            print('>>> Could not preload! Starting from scratch.')

    model.configure_optim(lr=hparams.lr)
    model = model.to(hparams.device)

    logger = WAndBLogger(
        info_log_interval=hparams.log_interval,
        model=model,
        hparams=hparams,
        type='video',
    )

    print(model.summary())

    tr = trange(hparams.its)

    for it in tr:
        batch = next(train_data_generator)
        train_loss, train_info = model.optim_step(batch)
        train_loss = train_loss.item()

        progress_description = \
            f'Loss: {train_loss:.6f}, Buff size: {len(train_data_generator)}'
        tr.set_description(progress_description)

        logger.log({'train_loss': train_loss})

        if it % hparams.log_interval == 0:
            logger.log_info(train_info, prefix='train')

            with torch.no_grad():
                val_batch = next(val_data_generator)
                val_loss, val_info = model.optim_step(val_batch)

                logger.log({'val_loss': val_loss})
                logger.log_info(val_info, prefix='val')

                num_example_rollouts = 5
                logger.log_images('example_val_rollout', [
                    get_example_rollout(val_info, i)
                    for i in range(num_example_rollouts)
                ])

                model.persist()


import argparse

defaults = dict(
    log_interval=500,
    frame_size=(32, 32),
    its=50_000,
    bs=32,
    lr=0.0002,
    device='cuda',
    model='rnn_deconvolve',
    env_name='TwoPlayerPong-32-v0',
    # env_name='CubeCrash-v0',
    # env_name='snek-rgb-16-v1',
    # env_name='CartPole-v1',
    # env_name='LunarLander-v2',
    precondition_size=3,
    max_seq_len=128,
    min_seq_len=35,
    moving_window_slices=None,
)

hparams = argparse.Namespace(**defaults)

if __name__ == '__main__':
    main(hparams)
