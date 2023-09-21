import os
ParqueFicticio_path = os.path.dirname(os.path.abspath("__file__")) + "/"
print(ParqueFicticio_path )
from ._parque_ficticio import ParqueFicticioSite  # nopep8
