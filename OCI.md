# SubCell Nahual OCI image

Build with `nix build .#oci-image` and load `result` with Podman or Docker. The
image is `nahual/subcell:local`, defaults to `tcp://0.0.0.0:5555`, and downloads
its selected encoder before starting the server.

```console
podman load < result
podman run --rm --device nvidia.com/gpu=all -p 5555:5555 \
  -v nahual-subcell-cache:/tmp/nahual nahual/subcell:local
```

Use Docker's `--gpus all`; CPU fallback is supported. Arguments are endpoint,
channel set, and model type, in that order. Defaults are `rybg` and
`mae_contrast_supcon_model`.

```console
pip install 'nahual==0.0.8' numpy
python oci/smoke_test.py
```
