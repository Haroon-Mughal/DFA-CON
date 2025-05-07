import torch
import torch.nn as nn
import torchvision.models as models
import timm
from models.resnet import ResNetWithHead
from transformers import CLIPModel, AutoModel, AutoImageProcessor

class NormalizedWrapper(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, x):
        with torch.no_grad():
            x = self.encoder(x)
            x = nn.functional.normalize(x, dim=-1)
        return x

class ViTDinoWrapper(nn.Module):
    def __init__(self, vit_model, mode="cls"):
        super().__init__()
        self.vit = vit_model
        self.mode = mode

    def forward(self, x):
        with torch.no_grad():
            tokens = self.vit.forward_features(x)  # shape: (B, N, D)
            cls_token = tokens[:, 0]               # shape: (B, D)
            patch_tokens = tokens[:, 1:]           # shape: (B, N-1, D)
            if self.mode == "cls":
                x = cls_token
            elif self.mode == "cls+gap":
                gap = patch_tokens.mean(dim=1)
                x = torch.cat([cls_token, gap], dim=1)  # shape: (B, 2D)
            else:
                raise ValueError(f"Unsupported ViT-DINO mode: {self.mode}")
            return nn.functional.normalize(x, dim=-1)

class DinoV2Wrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        with torch.no_grad():
            outputs = self.model(pixel_values=x)
            cls_token = outputs.last_hidden_state[:, 0]
            return nn.functional.normalize(cls_token, dim=-1)

def load_embedding_model(config: dict):
    name = config["model_name"].lower()
    device = config["device"]
    use_projection = config["use_projection"]
    model_path = config.get("model_path", None)
    feature_dim = config.get("feature_dim", 128)
    head_type = config.get("head_type", "mlp")
    vit_mode = config.get("vit_mode", "cls")

    if name == "dfa_con_rn":
        model = ResNetWithHead(head_type=head_type, feature_dim=feature_dim).to(device)
        if model_path is None:
            raise ValueError("Model path must be provided for DFA-Con.")
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        return model if use_projection else NormalizedWrapper(model.encoder)

    elif name == "resnet50_imagenet":
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT).to(device)
        model.fc = nn.Identity()
        model.eval()
        return NormalizedWrapper(model)

    elif name == "clip_vitb16":
        clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16").vision_model
        class CLIPWrapper(nn.Module):
            def __init__(self, clip_encoder):
                super().__init__()
                self.encoder = clip_encoder

            def forward(self, x):
                outputs = self.encoder(pixel_values=x)
                x = outputs.pooler_output
                return nn.functional.normalize(x, dim=-1)

        return CLIPWrapper(clip_model).to(device)

    elif name == "vit_dino":
        model = timm.create_model('vit_base_patch16_224', pretrained=True)
        model.head = nn.Identity()
        model.eval()
        return ViTDinoWrapper(model.to(device), mode=vit_mode)

    elif name == "dinov2_vitl14":
        processor = AutoImageProcessor.from_pretrained("facebook/dinov2-large")
        model = AutoModel.from_pretrained("facebook/dinov2-large").to(device)
        model.eval()
        return DinoV2Wrapper(model), processor

    else:
        raise ValueError(f"Unsupported embedding model: {name}")
