import sys
from pathlib import Path

# try:
#     #from ..skmap import loader
#     #from ..skmap import skmap_utils as sb_utils
#     from .. import skmap_bindings
#     print('Imported from package')
# except ImportError:

repository_root = Path(__file__).parent.parent.resolve()
print(f'Importing from local path: {repository_root}')
sys.path.insert(0,repository_root.as_posix())
import skmap_bindings
import skmap
from skmap.loader import warp_tile
# print('Imported from local path')
