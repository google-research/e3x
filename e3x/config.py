# Copyright 2024 The e3x Authors.
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

"""Global configuration options for e3x."""

from typing import Optional
from etils import epath
from .so3._normalization import check_normalization_is_valid
from .so3._normalization import Normalization


class Config:
  """Static class for storing global configuration options for e3x.

  Attributes:
    cartesian_order: Whether operations assume irreps are stored in Cartesian
      order or not by default.
    use_fused_tensor: Whether modules that internally need tensor products use
      :class:`FusedTensor <e3x.nn.modules.FusedTensor>` or :class:`Tensor
      <e3x.nn.modules.Tensor>` for computing the tensor product by default.
    normalization: Which normalization is assumed for the spherical harmonics by
      default.
    spherical_harmonics_cache: Path to disk cache with coefficients for
      computing spherical harmonics.
    clebsch_gordan_cache: Path to disk cache with Clebsch-Gordan coefficients.
    wigner_d_cache: Path to disk cache with coefficients for converting rotation
      matrices to Wigner-D matrices.
    tensor_conversion_cache: Path to disk cache with tensor conversion
      coefficients.
  """

  cartesian_order: bool = True
  use_fused_tensor: bool = False
  normalization: Normalization = 'racah'
  spherical_harmonics_cache: Optional[epath.Path] = None
  clebsch_gordan_cache: Optional[epath.Path] = None
  wigner_d_cache: Optional[epath.Path] = None
  tensor_conversion_cache: Optional[epath.Path] = None

  @staticmethod
  def set_cartesian_order(cartesian_order: bool = True) -> None:
    """Sets the value of Config.cartesian_order.

    Args:
      cartesian_order: New value for Config.cartesian_order.
    """
    Config.cartesian_order = cartesian_order

  @staticmethod
  def set_use_fused_tensor(use_fused_tensor: bool = False) -> None:
    """Sets the value of Config.use_fused_tensor.

    Args:
      use_fused_tensor: New value for Config.use_fused_tensor.
    """
    Config.use_fused_tensor = use_fused_tensor

  @staticmethod
  def set_normalization(normalization: Normalization = 'racah') -> None:
    """Sets the value of Config.normalization.

    Args:
      normalization: New value for Config.normalization.

    Raises:
      ValueError: If ``normalization`` has an invalid value.
    """
    check_normalization_is_valid(normalization)
    Config.normalization = normalization

  @staticmethod
  def set_spherical_harmonics_cache(
      path: Optional[epath.PathLike] = None,
  ) -> None:
    """Sets the value of Config.spherical_harmonics_cache.

    Args:
      path: Path to disk cache for saving/loading spherical harmonics
        coefficients.
    """
    Config.spherical_harmonics_cache = (
        epath.Path(path) if path is not None else None
    )

  @staticmethod
  def set_clebsch_gordan_cache(path: Optional[epath.PathLike] = None) -> None:
    """Sets the value of Config.clebsch_gordan_cache.

    Args:
      path: Path to disk cache for saving/loading Clebsch-Gordan coefficients.
    """
    Config.clebsch_gordan_cache = epath.Path(path) if path is not None else None

  @staticmethod
  def set_wigner_d_cache(path: Optional[epath.PathLike] = None) -> None:
    """Sets the value of Config.wigner_d_cache.

    Args:
      path: Path to disk cache for saving/loading coefficients for converting
        rotation matrices to Wigner-D matrices.
    """
    Config.wigner_d_cache = epath.Path(path) if path is not None else None

  @staticmethod
  def set_tensor_conversion_cache(
      path: Optional[epath.PathLike] = None,
  ) -> None:
    """Sets the value of Config.tensor_conversion_cache.

    Args:
      path: Path to disk cache for saving/loading tensor conversion
        coefficients.
    """
    Config.tensor_conversion_cache = (
        epath.Path(path) if path is not None else None
    )
