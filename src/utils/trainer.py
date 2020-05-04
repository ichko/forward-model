from tqdm.auto import tqdm, trange


def fit_generator(model, data_generator, its, logger, persist_frequency):
    tr = trange(its)

    for it in tr:
        batch = next(data_generator)
        info = model.optim_step(batch)

        loss = info['loss'].item()
        tr.set_description(f'Loss: {loss}')

        logger.log_info(info)

        if it % persist_frequency == 0:
            model.persist()
