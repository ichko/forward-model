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
