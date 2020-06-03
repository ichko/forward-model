# Forward model

Experiments and models for my masters thesis on learning environment dynamics from observations.

## Notes and tasks

- [Profiling code](https://toucantoco.com/en/tech-blog/tech/python-performance-optimization)

  - `pip install profiling`
  - `profiling live-profile -m src.pipelines.rnn -- --debug`

- General stuff

  - [x] Mask out empty (padded) frames after rollout has finished. [See here.](https://www.kdnuggets.com/2018/06/taming-lstms-variable-sized-mini-batches-pytorch.html)
  - [ ] Label smoothing. Do I actually want that?

- Models

  - [x] RNN Deconvolution Baseline
  - [x] Learn frame transformations
    - Instead of compressing the state like the RNN does
    - Action + Precondition (last few frames) -> transformation vector T
    - Use T to transform the current frame to the future frame
    - [x] Play rollout of frame transformations - results in wandb look promising

- Notes

  - 04.06.2020
    - [BUGFIX] Found major bug in RNN models - the pred frames and true frames were not aligned, the model was trying to predict the present from the present
    - [BUGFIX] TimeDistributed (decorator) module was not holding the wrapped module in it's state resulting in the parameters of the wrapped module not being part of the overall model, resulting in the model not being able to be trined. (Took quite some time)
    - [FEATURE] Implemented `generic multiprocessing` function spawner and `random agent rollout generator` that leads to newer rollouts in the training buffer faster. Hopefully this can reduce over-fitting.
