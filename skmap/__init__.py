from abc import ABC, abstractmethod

from skmap.misc import ttprint


class SKMapBase(ABC):
    def _verbose(self, *args, **kwargs) -> None:
        if self.verbose:
            ttprint(*args, **kwargs)


class SKMapRunner(SKMapBase, ABC):
    def __init__(
        self,
        verbose: bool = True,
    ) -> None:
        self.verbose = verbose

    @abstractmethod
    def run(self, data, outname: str):
        pass


class SKMapGroupRunner(SKMapBase, ABC):
    def __init__(self, verbose: bool = True, temporal: bool = False) -> None:
        self.verbose = verbose
        self.temporal = temporal

        temporal: bool = False

    @abstractmethod
    def run(self, data, group: str, outname: str) -> None:
        pass
