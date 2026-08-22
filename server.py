"""Model deployment for SubCell https://github.com/afermg/SubCellPortable."""

import sys
from functools import partial

import numpy
import pynng
import torch
import transformers
import trio
from loguru import logger
from nahual.preprocess import pad_channel_dim, validate_input_shape
from nahual.server import responder

from process import setup_model

# We will use pre-existing information to enforce guardrails on the input data
# model -> (expected #channels, mandated shape of yx)
guardrail_shapes = {
    "mae_contrast_supcon_model": (4, 16),  # TODO relax constraints?
}


def setup(model_type: str, model_channels: str, **kwargs) -> dict:
    # Some default values
    device = kwargs.get("device", 0)

    # Some default values
    setup_defaults = dict(
        model_type="mae_contrast_supcon_model",
        model_channels="rybg",
        device=torch.device(device),
    )
    execution_defaults = dict()

    setup_kwargs = kwargs.get("setup_kwargs", {})
    execution_kwargs = kwargs.get("execution_kwargs", {})

    # Define parameters by combining defaults and non-defaults
    setup_params = {**setup_defaults, **setup_kwargs}
    execution_params = {**execution_defaults, **execution_kwargs}

    device = setup_params.pop("device")
    # Load model instance
    print("before model loading")
    model = setup_model(**setup_params)
    print("model loading passed")
    model = model.to(device)
    # model.eval() gets done within setup_model

    expected_nchannels, yx_shape = guardrail_shapes[model_type]
    execution_params["expected_nchannels"] = expected_nchannels
    execution_params["expected_yx"] = yx_shape

    # Generate a json-encodable dictionary to send back to the client
    serializable_params = {
        name: {k: str(v) for k, v in d.items()}
        for name, d in zip(("setup", "execution"), (setup_params, execution_params))
    }

    # "Freeze" model in-place
    processor = partial(process_pixels, model=model, device=device, **execution_params)
    return processor, serializable_params


async def main():
    """Main function for the asynchronous server.

    This function sets up a nng connection using pynng and starts a nursery to handle
    incoming requests asynchronously.

    Parameters
    ----------
    address : str
        The network address to listen on.

    Returns
    -------
    None
    """

    with pynng.Rep0(listen=address, recv_timeout=300_000) as sock:
        print(f"Pretrained SubCell server listening on {address}")
        async with trio.open_nursery() as nursery:
            responder_curried = partial(responder, setup=setup)
            nursery.start_soon(responder_curried, sock)


@torch.inference_mode()
def process_pixels(
    pixels: numpy.ndarray,
    model,
    expected_yx: tuple[int],
    expected_nchannels: int,
    device: int,
) -> numpy.ndarray:
    """Apply a pretrained model. We pass arguments that encode the necessary input shapes and number of channels to pad. We will valudate the yx dimensions and pad the channel dimension with zeros.

    Input contract (caller side)
    ----------------------------
    pixels : NCZYX float32 in [0, 1]; H, W divisible by 16; Z=1.
        Exactly 4 channels in **rybg slot order** — the channels the SubCell
        paper trained on:
            r → Microtubules
            y → ER
            b → Nucleus
            g → Protein-of-interest
        For Cell Painting that maps to [AGP, ER, DNA, Mito] (AGP stains
        actin+golgi+plasma-membrane, used as a microtubule-like proxy).
        Pass already-clipped pixels in [0, 1] — do NOT z-score; the model
        was trained on linearly-scaled raw intensities and a z-score collapses
        the embedding to near-zero cosine vs. the published output.

    Server-side normalization (applied here)
    ----------------------------------------
    None — ``model(pixels)`` runs the published forward directly.

    Output
    ------
    (N, 1536) — pool_op embedding (4 channels × 384-d each).
    """

    _, input_channels, _, *input_yx = pixels.shape

    validate_input_shape(input_yx, expected_yx)

    pixels = pad_channel_dim(pixels, expected_nchannels).copy()

    pixels_torch = torch.from_numpy(pixels).float().to(device)

    embeddings = model(pixels_torch)
    # SubCell output: (N, 1536)
    embeddings_np = embeddings.pool_op.cpu().numpy()

    return embeddings_np


if __name__ == "__main__":
    # address = "ipc:///tmp/subcell.ipc"
    address = sys.argv[1]

    logger.add(address.split("/")[-1])

    try:
        trio.run(main)
    except KeyboardInterrupt:
        # that's the way the program *should* end
        pass
