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

    @abstractmethod
    def run(self, data, outname: str):
        """Process ``data`` and write the result under ``outname``."""
        pass


class SKMapGroupRunner(SKMapBase, ABC):
    """Abstract runner that processes data grouped by ``group`` (e.g. by year)."""

    def __init__(self, verbose: bool = True, temporal: bool = False) -> None:
        self.verbose = verbose
        self.temporal = temporal

        temporal: bool = False

    @abstractmethod
    def run(self, data, group: str, outname: str) -> None:
        """Process ``data`` for ``group`` and write the result under ``outname``."""
        pass
