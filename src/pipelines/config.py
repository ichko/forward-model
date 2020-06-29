import argparse


def get_hparams(config_id):
    defaults = dict(
        log_interval=500,
        frame_size=(32, 32),
        its=50_000,
        bs=32,
        lr=0.0002,
        device='cuda',
        # env_name='TwoPlayerPong-32-v0',
        # env_name='CubeCrash-v0',
        # env_name='snek-rgb-16-v1',
        # env_name='CartPole-v1',
        # env_name='LunarLander-v2',
        precondition_size=3,
        max_seq_len=128,
        min_seq_len=35,
        moving_window_slices=None,
    )

    configs = dict(
        ## RNN Based
        rnn_deconvolve_cube=dict(
            env_name='CubeCrash-v0',
            lr=0.001,
            model='rnn_deconvolve',
            bs=64,
            precondition_size=1,
            max_seq_len=32,
            min_seq_len=2,
            moving_window_slices=None,
        ),
        rnn_deconvolve_pong=dict(
            env_name='TwoPlayerPong-32-v0',
            lr=0.001,
            model='rnn_deconvolve',
            bs=32,
            precondition_size=3,
            max_seq_len=128,
            min_seq_len=64,
            moving_window_slices=None,
        ),
        rnn_dense_cube=dict(
            env_name='CubeCrash-v0',
            lr=0.001,
            model='rnn_dense',
            bs=64,
            precondition_size=1,
            max_seq_len=32,
            min_seq_len=2,
            moving_window_slices=None,
        ),
        rnn_dense_pong=dict(
            env_name='TwoPlayerPong-32-v0',
            lr=0.001,
            model='rnn_dense',
            bs=32,
            precondition_size=2,
            max_seq_len=128,
            min_seq_len=64,
            moving_window_slices=None,
        ),

        ## Spatial Transformer
        rnn_spatial_transformer_cube=dict(
            env_name='CubeCrash-v0',
            lr=0.001,
            model='rnn_spatial_transformer',
            bs=16,
            precondition_size=1,
            max_seq_len=32,
            min_seq_len=2,
            moving_window_slices=None,
        ),
        rnn_spatial_transformer_pong=dict(
            env_name='TwoPlayerPong-32-v0',
            lr=0.001,
            model='rnn_spatial_transformer',
            bs=16,
            precondition_size=2,
            max_seq_len=128,
            min_seq_len=64,
            moving_window_slices=None,
        ),

        ## Frame transformers
        frame_transformer_dense_pong=dict(
            env_name='TwoPlayerPong-32-v0',
            lr=0.001,
            model='frame_transformer_dense',
            bs=16,
            precondition_size=2,
            max_seq_len=128,
            min_seq_len=64,
            moving_window_slices=3,
        ),
        frame_transformer_dense_cube=dict(
            env_name='CubeCrash-v0',
            lr=0.001,
            model='frame_transformer_dense',
            bs=256,
            precondition_size=1,
            max_seq_len=32,
            min_seq_len=2,
            moving_window_slices=2,
        ),
    )

    # config = configs['rnn_deconvolve']
    # config = configs['rnn_dense']
    # config = configs['rnn_spatial_transformer']
    # config = configs['frame_transformer_dense']
    config = configs[config_id]

    config_dict = {**defaults, **config}
    hparams = argparse.Namespace(**config_dict)

    return hparams, config_dict