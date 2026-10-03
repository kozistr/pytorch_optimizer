import pytest
import torch

from pytorch_optimizer.optimizer.galore_utils import GaLoreProjector


class TestGaLoreProjector:
    """Tests for GaLore projector methods."""

    @pytest.fixture
    def sample_tensor(self):
        return torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)

    def test_invalid_projection_type_project_with_ortho(self, sample_tensor):
        invalid_galore = GaLoreProjector(projection_type='invalid')
        invalid_galore.ortho_matrix = sample_tensor
        with pytest.raises(NotImplementedError):
            invalid_galore.project(sample_tensor, 1)

    def test_invalid_projection_type_project(self, sample_tensor):
        with pytest.raises(NotImplementedError):
            GaLoreProjector(projection_type='invalid').project(sample_tensor, 1)

    def test_invalid_projection_type_project_back(self, sample_tensor):
        with pytest.raises(NotImplementedError):
            GaLoreProjector(projection_type='invalid').project_back(sample_tensor)

    def test_full_projection_project_validation(self, sample_tensor):
        full_projector = GaLoreProjector(projection_type='full')
        full_projector.ortho_matrix = sample_tensor
        with pytest.raises(ValueError):
            full_projector.project(sample_tensor, 1)

    def test_full_projection_project_back_validation(self, sample_tensor):
        full_projector = GaLoreProjector(projection_type='full')
        full_projector.ortho_matrix = sample_tensor
        with pytest.raises(ValueError):
            full_projector.project_back(sample_tensor)

    def test_left_projection_with_random_matrix(self, sample_tensor):
        projector = GaLoreProjector(projection_type='left', rank=1)
        projector.get_orthogonal_matrix(sample_tensor, projection_type='left', from_random_matrix=True)

    def test_left_projection_without_rank(self, sample_tensor):
        projector = GaLoreProjector(projection_type='left', rank=None)
        with pytest.raises(TypeError):
            projector.get_orthogonal_matrix(sample_tensor, projection_type='left', from_random_matrix=True)

    def test_std_projection_invalid(self, sample_tensor):
        projector = GaLoreProjector(projection_type='std', rank=1)
        with pytest.raises(ValueError):
            projector.get_orthogonal_matrix(sample_tensor, projection_type='std')
