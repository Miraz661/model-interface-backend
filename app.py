import torch
import timm

def load_model(weights_path='deit_model_b16_02.pth', num_classes=2):
    model = timm.create_model('deit_base_patch16_224', pretrained=False, num_classes=num_classes)
    
    # Load full checkpoint
    checkpoint = torch.load(weights_path, map_location=torch.device('cpu'))

    # Extract only the model state
    if 'model_state' in checkpoint:
        model.load_state_dict(checkpoint['model_state'])
    else:
        model.load_state_dict(checkpoint)  # fallback if it's already just weights

    model.eval()
    return model
