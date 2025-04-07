import torch
import torch.nn as nn
import timm

class XrayModel(nn.Module):
    def __init__(self, num_classes=14, model_name="efficientnet_b0", pretrained=True):
        super(XrayModel, self).__init__()

        self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0)

        with torch.no_grad():
            dummy_input = torch.zeros(1, 3, 384, 384)
            features = self.backbone(dummy_input)
            n_features = features.shape[1]
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.3), #overfitting is handled
            nn.Linear(n_features, num_classes)
        )
    
    def forward(self, x):
        x = self.backbone(x)
        return self.classifier(x)

def load_model(model_path=None):
    model = XrayModel(model_name="efficientnet_b0")
    if model_path and os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
    model.eval()
    return model


DISEASE_LABELS = [
    'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration',
    'Mass', 'Nodule', 'Pneumonia', 'Pneumothorax', 'Consolidation',
    'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia'
]