# Copyright 2024 The HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""HATTENTION configuration"""

import sys
from pathlib import Path

# Add FLA (Flash-Linear-Attention) to path for dynamic imports
from hattention.config import get_fla_base_path
_fla_base = Path(get_fla_base_path()).parent  # Remove trailing slash
if str(_fla_base) not in sys.path:
    sys.path.insert(0, str(_fla_base))

from fla.models.mamba2 import Mamba2Config


class HAttentionConfig(Mamba2Config):

    model_type = "hattention"
