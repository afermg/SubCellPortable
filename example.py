# Load model directly
import numpy
import torch

from process import run_inference, setup_model

model = setup_model("rybg", "mae_contrast_supcon_model")
# %%
tile_size = 256
data = numpy.random.random_sample((1, 4, tile_size, tile_size))
torch_tensor = torch.from_numpy(data).float().cuda()
with torch.no_grad():
    result = model(torch_tensor)
# result = run_inference("rybg", "mae_contrast_supcon_model", 0, torch_tensor)
print([x.shape for x in result])
# torch.Size([1, 384])
np_val = result.cpu().detach().numpy()
