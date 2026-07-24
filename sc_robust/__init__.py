from ._version import __version__
from .compatibility import *  # noqa: F401,F403
from .provenance import *  # noqa: F401,F403

from sc_robust.sc_robust import *
from sc_robust.count_split_adapter import *  # noqa: F401,F403
from sc_robust.de import *  # noqa: F401,F403

# Gene module discovery (spearman.hdf5 meta-analysis helpers)
from sc_robust.gene_modules import *  # noqa: F401,F403
