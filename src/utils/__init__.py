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


def add_virtual_display():
    # Necessary to display cartpole and other envs headlessly
    # https://stackoverflow.com/a/47968021
    from pyvirtualdisplay.smartdisplay import SmartDisplay

    display = SmartDisplay(visible=0, size=(1400, 900))
    display.start()

    # print(os.environ['DISPLAY'])


def try_colored_traceback():
    try:
        import colored_traceback
        colored_traceback.add_hook()
    except ImportError:
        pass
