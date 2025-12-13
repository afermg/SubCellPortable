# Load model directly
import numpy
import torch

from process import setup_model

model = setup_model("rybg", "mae_contrast_supcon_model")
tile_size = 256
data = numpy.random.random_sample((2, 4, tile_size, tile_size))
torch_tensor = torch.from_numpy(data).float().cuda()
with torch.no_grad():
    result = model(torch_tensor).pool_op.cpu().numpy()
print(result.shape)
# (2, 1536)
