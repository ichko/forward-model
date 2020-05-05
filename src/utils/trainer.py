from tqdm.auto import tqdm, trange
import torch


def fit_generator(
    model,
    train_data_generator,
    val_data_generator,
    its,
    logger,
    persist_frequency,
    log_info_interval,
):
    tr = trange(its)

    for it in tr:
        batch = next(train_data_generator)
        train_loss, train_info = model.optim_step(batch)

        train_loss = train_loss.item()
        tr.set_description(f'Loss: {train_loss}')
        logger.log({'train_loss': train_loss})

        if it % log_info_interval == 0:
            logger.log_info(train_info, prefix='train')

            with torch.no_grad():
                val_batch = next(val_data_generator)
                val_loss, val_info = model.optim_step(val_batch)

                logger.log({'val_loss': val_loss})
                logger.log_info(val_info, prefix='val')

        if it % persist_frequency == 0:
            model.persist()
