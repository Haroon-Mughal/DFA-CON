import torch.nn as nn
import torchvision.models as models

class ResNetWithHead(nn.Module):
    """
    ResNet-50 backbone with a projection head for contrastive learning.
    """

    def __init__(self, head_type='mlp', feature_dim=128):
        """
        Args:
            head_type (str): Type of projection head: 'linear' or 'mlp'.
            feature_dim (int): Output dimension of the projection head.
        """
        super(ResNetWithHead, self).__init__()

        # Load ResNet-50 backbone (pretrained on ImageNet)
        self.encoder = models.resnet50(pretrained=True)

        # Extract the number of features before the final FC layer
        num_ftrs = self.encoder.fc.in_features

        # Remove the original classification layer
        self.encoder.fc = nn.Identity()

        # Add projection head on top of the encoder
        if head_type == 'linear':
            self.head = nn.Linear(num_ftrs, feature_dim)
        elif head_type == 'mlp':
            self.head = nn.Sequential(
                nn.Linear(num_ftrs, num_ftrs),
                nn.ReLU(),
                nn.Linear(num_ftrs, feature_dim)
            )
        else:
            raise ValueError("head_type must be 'linear' or 'mlp'")

    def forward(self, x):
        """
        Forward pass: returns encoded + projected feature vectors
        """
        x = self.encoder(x)  # ResNet feature extractor
        x = self.head(x)     # Projection head
        return x
