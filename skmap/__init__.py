from abc import ABC, abstractmethod

from skmap.misc import ttprint


class SKMapBase(ABC):
    """Abstract base class for all skmap runners, providing a ``verbose`` helper."""

    def _verbose(self, *args, **kwargs) -> None:
        if self.verbose:
            ttprint(*args, **kwargs)


class SKMapRunner(SKMapBase, ABC):
    """Abstract runner that processes a single data input into one named output."""

    def __init__(
        self,
        verbose: bool = True,
    ) -> None:
        self.verbose = verbose
        from skmap.compute import NumpyBackend

        self.backend = NumpyBackend()

    @abstractmethod
    def run(self, data, outname: str):
        """Process ``data`` and write the result under ``outname``."""
        pass


class SKMapGroupRunner(SKMapBase, ABC):
    """Abstract runner that processes data partitioned by ``feature``."""

    def __init__(self, verbose: bool = True) -> None:
        self.verbose = verbose
        from skmap.compute import NumpyBackend

        self.backend = NumpyBackend()

    @abstractmethod
    def run(self, data, feature: str, outname: str) -> None:
        """Process ``data`` for ``feature`` and write the result under ``outname``."""
        pass
