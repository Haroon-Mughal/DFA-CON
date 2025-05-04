import torch
import torch.nn as nn
import torch.nn.functional as F

class SupConLoss(nn.Module):
    """
    Supervised Contrastive Loss from:
    "Supervised Contrastive Learning" (Khosla et al., NeurIPS 2020)

    Encourages embeddings with the same label (group ID) to be close,
    and those with different labels to be apart.
    """

    def __init__(self, temperature: float = 0.07):
        """
        Args:
            temperature: Scaling factor for similarities (default 0.07 from the paper)
        """
        super(SupConLoss, self).__init__()
        self.temperature = temperature

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: Tensor of shape [N, D] — N images in batch, each with D-dim embedding
            labels: Tensor of shape [N] — group ID for each embedding

        Returns:
            Scalar loss (float)
        """
        device = features.device
        N = features.shape[0]

        # Normalize embeddings: ensures cosine similarity
        features = F.normalize(features, p=2, dim=1)

        # Compute full cosine similarity matrix
        sim_matrix = torch.matmul(features, features.T) / self.temperature

        # Subtract max for numerical stability (optional but recommended)
        sim_matrix = sim_matrix - sim_matrix.max(dim=1, keepdim=True)[0].detach()

        # Mask: 1 where label[i] == label[j] and i != j
        labels = labels.view(-1, 1)  # [N, 1]
        mask = torch.eq(labels, labels.T).float().to(device)
        mask = mask.fill_diagonal_(0)  # remove self-pairs

        # Exponentiate sim matrix
        exp_sim = torch.exp(sim_matrix)

        # Denominator: sum over all j ≠ i
        denom = exp_sim.sum(dim=1, keepdim=True)

        # Log prob of positives
        log_prob = sim_matrix - torch.log(denom + 1e-12)

        # Masked mean log-prob for positive pairs only
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-12)

        # Final loss
        loss = -mean_log_prob_pos.mean()

        return loss
