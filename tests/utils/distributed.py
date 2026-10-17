from collections.abc import Callable
from datetime import timedelta
from pathlib import Path

import torch.distributed as dist
import torch.multiprocessing as mp


def _run_worker(rank: int, rendezvous: str, worker: Callable, args: tuple) -> None:
    dist.init_process_group('gloo', init_method=rendezvous, rank=rank, world_size=2, timeout=timedelta(seconds=30))
    try:
        worker(rank, *args)
    finally:
        dist.destroy_process_group()


def run_distributed(worker: Callable, rendezvous: Path, *args) -> None:
    uri = rendezvous.resolve().as_uri()
    process = mp.get_context('spawn').Process(target=_run_worker, args=(1, uri, worker, args))
    process.start()
    try:
        _run_worker(0, uri, worker, args)
    except BaseException:
        process.terminate()
        raise
    finally:
        process.join(timeout=45)
        if process.is_alive():
            process.terminate()
            process.join()
            raise RuntimeError('Distributed test worker did not exit.')
    assert process.exitcode == 0, f'Distributed test worker exited with code {process.exitcode}.'
