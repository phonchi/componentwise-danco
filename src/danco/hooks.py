"""Layer selection and activation capture for the CNN experiments.

Ansuini et al. measure intrinsic dimension at a handful of *checkpoints* along a
network rather than at every layer, and which layers those are is
architecture-specific.  Two conventions are used:

* **feature/classifier nets** (AlexNet, VGG): every ``MaxPool2d`` in
  ``model.features``, then every ``Linear`` in ``model.classifier``;
* **ResNets**: ``maxpool``, the four stages, ``avgpool``, ``fc``.

Both prepend a pseudo-layer ``'input'`` -- the flattened image itself, the
``D = 150528`` case that sets the memory budget for the whole run.

``depth`` is the running count of convolutions/linears *up to* that checkpoint,
which is the x-axis of the hump plot.  It is not the index of the checkpoint,
and the two are easy to confuse: AlexNet's three max-pools sit at depths 1, 2
and 5.

The capture side is one forward pass per batch with a hook on every checkpoint
at once, rather than one pass per layer.  Activations are flattened to
``(batch, -1)`` and moved to the CPU as float32 immediately; keeping a VGG's
early feature maps on an 8 GB card is not an option.
"""

import time

import numpy as np

__all__ = [
    "WEIGHT_ENUMS",
    "build_pretrained_model",
    "layer_checkpoints",
    "extract_activations",
]


# torchvision moved from `pretrained=True` to weight enums; the mapping is not
# derivable from the arch name alone (``vgg16_bn`` -> ``VGG16_BN_Weights``).
WEIGHT_ENUMS = {
    "alexnet": "AlexNet_Weights",
    "vgg11": "VGG11_Weights",
    "vgg13": "VGG13_Weights",
    "vgg16": "VGG16_Weights",
    "vgg19": "VGG19_Weights",
    "vgg11_bn": "VGG11_BN_Weights",
    "vgg13_bn": "VGG13_BN_Weights",
    "vgg16_bn": "VGG16_BN_Weights",
    "vgg19_bn": "VGG19_BN_Weights",
    "resnet18": "ResNet18_Weights",
    "resnet34": "ResNet34_Weights",
    "resnet50": "ResNet50_Weights",
    "resnet101": "ResNet101_Weights",
    "resnet152": "ResNet152_Weights",
}


def build_pretrained_model(arch, pretrained=True):
    """Return ``(model, weight_name)`` for a torchvision architecture."""
    import torchvision

    try:
        constructor = getattr(torchvision.models, arch)
    except AttributeError as exc:
        raise ValueError(f"unknown torchvision architecture {arch!r}") from exc
    if not pretrained:
        return constructor(weights=None), "None"
    if arch not in WEIGHT_ENUMS:
        raise ValueError(
            f"no pretrained-weight enum recorded for {arch!r}; known: "
            f"{sorted(WEIGHT_ENUMS)}"
        )
    weights = getattr(torchvision.models, WEIGHT_ENUMS[arch]).DEFAULT
    return constructor(weights=weights), str(weights)


def _features_checkpoints(model):
    """MaxPool2d in ``features`` + Linear in ``classifier`` (AlexNet, VGG)."""
    modules, names, depths = ["input"], ["input"], [0]
    count = 0
    for module in model.features:
        name = module.__class__.__name__
        if "Conv2d" in name or "Linear" in name:
            count += 1
        if "MaxPool2d" in name:
            modules.append(module)
            names.append("MaxPool2d")
            depths.append(count)
    for module in model.classifier:
        if "Linear" in module.__class__.__name__:
            modules.append(module)
            count += 1
            # +1: the depth convention counts the classifier's own position on
            # top of the running conv/linear count.
            depths.append(count + 1)
            names.append("Linear")
    return modules, names, np.array(depths)


def _stage_conv_count(stage):
    return sum(
        1
        for block in stage
        for child in block.children()
        if "Conv" in child.__class__.__name__
    )


def _resnet_checkpoints(model):
    modules, names, depths = ["input"], ["input"], [0]
    count = 1
    modules.append(model.maxpool)
    names.append("maxpool")
    depths.append(count)
    for i in range(1, 5):
        stage = getattr(model, f"layer{i}")
        count += _stage_conv_count(stage)
        modules.append(stage)
        names.append(f"layer{i}")
        depths.append(count)
    count += 1
    modules.append(model.avgpool)
    names.append("avgpool")
    depths.append(count)
    count += 1
    modules.append(model.fc)
    names.append("fc")
    depths.append(count)
    return modules, names, np.array(depths)


def layer_checkpoints(model, arch):
    """``(modules, names, depths)`` for the checkpoints of ``arch``.

    ``modules[0]`` is the string ``'input'``, not a module: the first
    "activation" is the flattened input image, which no hook can produce.
    """
    if "resnet" in arch:
        return _resnet_checkpoints(model)
    return _features_checkpoints(model)


def extract_activations(model, arch, loader, device=None):
    """Flattened activations at every checkpoint, in one pass per batch.

    Parameters
    ----------
    model : torch.nn.Module
        Already on ``device`` and in ``eval()`` mode.
    arch : str
        Selects the checkpoint convention (see :func:`layer_checkpoints`).
    loader : iterable
        Yields ``(inputs, targets)`` -- a ``DataLoader`` or anything shaped like
        one.
    device : torch.device or None
        Defaults to the device of the model's first parameter.

    Returns
    -------
    activations : list of ndarray
        One ``(n_samples, D_layer)`` float32 array per checkpoint.
    names, depths : list, ndarray
    profile : dict
        Wall-clock split, plus ``per_layer_shapes``.
    """
    import torch

    if device is None:
        device = next(model.parameters()).device
    modules, names, depths = layer_checkpoints(model, arch)
    targets = [(pos, m) for pos, m in enumerate(modules) if m != "input"]

    buckets = [[] for _ in modules]
    current = {}
    handles = []
    for pos, module in targets:
        def make_hook(p):
            def hook(_module, _input, output):
                current[p] = output
            return hook
        handles.append(module.register_forward_hook(make_hook(pos)))

    profile = dict(loader_wait_s=0.0, forward_s=0.0, cpu_copy_s=0.0,
                   concat_s=0.0, feature_extract_wall_s=0.0)
    wall0 = time.perf_counter()
    try:
        with torch.inference_mode():
            it = iter(loader)
            while True:
                t0 = time.perf_counter()
                try:
                    inputs, _ = next(it)
                except StopIteration:
                    break
                profile["loader_wait_s"] += time.perf_counter() - t0

                current.clear()
                t0 = time.perf_counter()
                model(inputs.to(device, non_blocking=True))
                if device.type == "cuda":
                    torch.cuda.synchronize()
                profile["forward_s"] += time.perf_counter() - t0

                t0 = time.perf_counter()
                n = inputs.shape[0]
                buckets[0].append(
                    inputs.reshape(n, -1).detach().cpu().to(torch.float32)
                )
                for pos, _ in targets:
                    if pos not in current:
                        raise RuntimeError(
                            f"no hook output for {arch} checkpoint {pos} "
                            f"({names[pos]}); the module is not on the forward "
                            "path for this input"
                        )
                    buckets[pos].append(
                        current[pos].reshape(n, -1).detach().cpu()
                        .to(torch.float32)
                    )
                profile["cpu_copy_s"] += time.perf_counter() - t0
    finally:
        for handle in handles:
            handle.remove()

    t0 = time.perf_counter()
    activations = [torch.cat(parts, dim=0).numpy() for parts in buckets]
    profile["concat_s"] = time.perf_counter() - t0
    profile["feature_extract_wall_s"] = time.perf_counter() - wall0
    profile["per_layer_shapes"] = [list(a.shape) for a in activations]
    return activations, names, depths, profile
