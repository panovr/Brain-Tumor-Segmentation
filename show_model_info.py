import bts.model as model
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 6
FILTER_LIST = [16,32,64,128,256]
unet_model = model.DynamicUNet(FILTER_LIST)
unet_model.summary(batch_size=BATCH_SIZE, device=device)
