import multiprocessing as mp
from collections import deque
import atexit


class MiltiprocessBuffer:
    def __init__(self, buffer_size, generator_func, num_processes):
        self.buffer = mp.Queue(maxsize=buffer_size)
        self.generator_func = generator_func
        self.lock = mp.Lock()
        self.processes = [
            mp.Process(target=self._run, args=(i, ))
            for i in range(num_processes)
        ]

        atexit.register(self.terminate)

    def start(self):
        for p in self.processes:
            p.start()

    def terminate(self):
        for p in self.processes:
            if p.is_alive:
                p.terminate()

    def try_pop(self):
        if self.buffer.empty(): return None
        return self.buffer.get()

    def pop(self):
        while True:
            if not self.buffer.empty():
                return self.buffer.get()

    def get(self, n):
        result = []
        while len(result) < n:
            if not self.buffer.empty():
                value = self.buffer.get()
                result.append(value)

        return result

    def _run(self, proc_id):
        generator = self.generator_func()
        while True:
            while not self.buffer.full():
                value = next(generator)
                with self.lock:
                    self.buffer.put(value)


if __name__ == '__main__':
    import random
    import time

    def gen_init():
        val = random.randint(0, 100)
        while True:
            time.sleep(1)
            val += 1
            yield val

    mpb = MiltiprocessBuffer(
        buffer_size=1000,
        generator_func=gen_init,
        num_processes=64,
    )

    mpb.start()

    while True:
        values = mpb.get(5)
        print(values)
