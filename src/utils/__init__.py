import os
import pickle


def persist(func, path, override=False):
    if not override and os.path.exists(path):
        with open(path, 'rb') as f:
            return pickle.load(f)

    result = func()
    with open(path, 'wb+') as f:
        pickle.dump(result, f)

    return result
