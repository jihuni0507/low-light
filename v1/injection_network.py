import torch
import torch.nn as nn


class ConditionedGaussianSHNet(nn.Module):
    """Learnable network that conditions Gaussian SH features on a CLIP vector.

    Input:
        gaussian_features: tensor shaped (N, K, C), typically from GaussianModel.get_features
        condition: tensor shaped (D,) or (B, D), where D is CLIP embedding dimension

    Output:
        updated_features: same shape as gaussian_features, with only SH feature values changed.
    """

    def __init__(
        self,
        condition_dim: int,
        sh_channels: int = 3,
        hidden_dim: int = 256,
        num_sh_coeffs: int | None = None,
    ):
        super().__init__()
        self.condition_dim = condition_dim
        self.sh_channels = sh_channels
        self.hidden_dim = hidden_dim
        self.num_sh_coeffs = num_sh_coeffs

        self.condition_mlp = nn.Sequential(
            nn.Linear(condition_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )

        self.gaussian_mlp = nn.Sequential(
            nn.Linear(sh_channels, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )

        self.feature_mixer = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )

        self.output_layer = nn.Linear(hidden_dim, sh_channels)

    def _prepare_condition(self, condition, num_gaussians: int):
        if condition is None:
            raise ValueError("condition vector must not be None")

        condition = condition.to(dtype=torch.float32)
        if condition.dim() == 1:
            condition = condition.unsqueeze(0)

        if condition.shape[-1] != self.condition_dim:
            raise ValueError(
                f"Expected condition dim {self.condition_dim}, got {condition.shape[-1]}"
            )

        if condition.shape[0] == 1 and num_gaussians > 1:
            condition = condition.expand(num_gaussians, -1)

        if condition.shape[0] != num_gaussians:
            raise ValueError(
                f"Condition batch size {condition.shape[0]} does not match gaussian count {num_gaussians}"
            )

        return condition

    def forward(self, gaussian_features: torch.Tensor, condition: torch.Tensor, return_delta: bool = False):
        """Condition Gaussian SH features with a CLIP vector.

        Args:
            gaussian_features: Tensor of shape (N, K, C) or (N, C)
            condition: Tensor of shape (D,) or (N, D)
            return_delta: if True, also return adjustment tensor

        Returns:
            updated_sh: same shape as gaussian_features
            delta_sh: optional, shape same as gaussian_features
        """
        if gaussian_features.dim() == 2:
            gaussian_features = gaussian_features.unsqueeze(1)

        if gaussian_features.shape[-1] != self.sh_channels:
            raise ValueError(
                f"Gaussian feature channel mismatch: expected {self.sh_channels}, got {gaussian_features.shape[-1]}"
            )

        num_gaussians = gaussian_features.shape[0]
        condition = self._prepare_condition(condition, num_gaussians)

        gaussian_emb = self.gaussian_mlp(gaussian_features)
        cond_emb = self.condition_mlp(condition)
        cond_emb = cond_emb.unsqueeze(1).expand(-1, gaussian_features.shape[1], -1)

        mixed = torch.cat([gaussian_emb, cond_emb], dim=-1)
        hidden = self.feature_mixer(mixed)
        delta_sh = self.output_layer(hidden)

        updated_sh = gaussian_features + delta_sh

        if return_delta:
            return updated_sh, delta_sh
        return updated_sh

    def inject_into_gaussian_model(self, gaussian_model, condition: torch.Tensor):
        """Apply the network directly to a GaussianModel instance.

        This keeps geometry parameters untouched and only updates SH features.
        """
        base_features = gaussian_model.get_features
        updated_features = self(base_features, condition)

        dc_features = updated_features[:, :1, :]
        rest_features = updated_features[:, 1:, :]
        gaussian_model.set_features(dc_features, rest_features)
        return updated_features


class GaussianSHInjectionNetwork(nn.Module):
    """Compatibility wrapper for future model code names.

    The current project imports `InjectionNetwork` or `ConditionedGaussianSHNet` depending on naming.
    """

    def __init__(self, condition_dim: int, sh_channels: int = 3, hidden_dim: int = 256):
        super().__init__()
        self.net = ConditionedGaussianSHNet(
            condition_dim=condition_dim,
            sh_channels=sh_channels,
            hidden_dim=hidden_dim,
        )

    def forward(self, gaussian_features: torch.Tensor, condition: torch.Tensor, return_delta: bool = False):
        return self.net(gaussian_features, condition, return_delta=return_delta)

    def inject_into_gaussian_model(self, gaussian_model, condition: torch.Tensor):
        return self.net.inject_into_gaussian_model(gaussian_model, condition)


InjectionNetwork = ConditionedGaussianSHNet
