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


def fit(model, data, epochs, logger, log_interval):
    train_data, val_data = data
    val_iter = iter(val_data)

    for e_id in tqdm(range(epochs)):

        tr = tqdm(enumerate(train_data))
        for i, batch in tr:
            train_loss, train_info = model.optim_step(batch)

            tr.set_description(f'Loss: {train_loss}')
            logger.log({
                'train_loss': train_loss,
                'epoch': e_id,
                'lr_scheduler': model.scheduler.get_lr()[0],
            })

            if i % log_interval == 0:
                logger.log_info(train_info, prefix='train')

                with torch.no_grad():
                    try:
                        batch = next(val_iter)
                    except StopIteration as _e:
                        val_iter = iter(val_data)
                        batch = next(val_iter)

                    val_loss, val_info = model.optim_step(batch)
                    logger.log({'val_loss': val_loss})
                    logger.log_info(val_info, prefix='val')

                    model.persist()

        # End of epoch
        model.scheduler.step()
