import torch
from PIL import Image
from torchvision import transforms


CHEXPERT_LABELS = ['Atelectasis',
    'Cardiomegaly',
    'Consolidation',
    'Edema',
    'Enlarged Cardiomediastinum',
    'Fracture',
    'Lung Lesion',
    'Lung Opacity',
    'Pleural Effusion',
    'Pleural Other',
    'Pneumonia',
    'Pneumothorax',
    'Support Devices',
]


eval_transforms = transforms.Compose([
    ##### YOUR CODE START #####


    ##### YOUR CODE END #####
])


def load_model():
    """
    Returns:
        model (torch.nn.Module): A PyTorch model
            - forward input: An image tensor preprocessed using `eval_transforms`
            - forward output: A logits tensor of shape `(batch_size, num_classes)`,
              where the class order corresponds to the labels defined in `CHEXPERT_LABELS`
    """
    ##### YOUR CODE START #####




    ##### YOUR CODE END #####
    return model

