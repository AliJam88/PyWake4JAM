import os
ParqueFicticio_path = os.path.dirname(os.path.abspath("__file__")) + "/"

#ParqueFicticio_path = os.path.join(os.path.dirname(__file__), 'AJ_ParqueFicticio/')
print(ParqueFicticio_path )

from ._parque_ficticio import ParqueFicticioSite  # nopep8
