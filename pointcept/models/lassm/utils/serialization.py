import torch
from pointcept.models.utils.serialization.z_order import xyz2key as z_order_encode_
from pointcept.models.utils.serialization.hilbert import encode as hilbert_encode_


def serialization(coord, order="z", depth=None):
    """
    Point Cloud Serialization

    relay on ["grid_coord" or "coord" + "grid_size", "batch", "feat"]
    """
    coord = coord - coord.min(0)[0]
    if depth is None:
        # Adaptive measure the depth of serialization cube (length = 2 ^ depth)
        try:
            depth = int(coord.max()).bit_length()
        except ValueError:
            print("coord shape: ", coord.shape)
            depth = 16
    # Maximum bit length for serialization code is 63 (int64)
    assert depth * 3 + int(1).bit_length() <= 63
    assert depth <= 16

    # The serialization codes are arranged as following structures:
    # [Order1 ([n]),
    #  Order2 ([n]),
    #   ...
    #  OrderN ([n])] (k, n)
    code = [
        encode(coord, 1, depth, order=order_) for order_ in order
    ]
    code = torch.stack(code)
    order = torch.argsort(code)
    inverse = torch.zeros_like(order).scatter_(
        dim=1,
        index=order,
        src=torch.arange(0, code.shape[1], device=order.device).repeat(
            code.shape[0], 1
        ),
    )
    return code, order, inverse


@torch.inference_mode()
def encode(grid_coord, batch=None, depth=16, order="z"):
    assert order in {"z", "z-trans", "hilbert", "hilbert-trans"}
    if order == "z":
        code = z_order_encode(grid_coord, depth=depth)
    elif order == "z-trans":
        code = z_order_encode(grid_coord[:, [1, 0, 2]], depth=depth)
    elif order == "hilbert":
        code = hilbert_encode(grid_coord, depth=depth)
    elif order == "hilbert-trans":
        code = hilbert_encode(grid_coord[:, [1, 0, 2]], depth=depth)
    else:
        raise NotImplementedError
    if batch is not None:
        # batch = batch.long()
        code = batch << depth * 3 | code
    return code


def z_order_encode(grid_coord: torch.Tensor, depth: int = 16):
    x, y, z = grid_coord[:, 0].long(), grid_coord[:, 1].long(), grid_coord[:, 2].long()
    # we block the support to batch, maintain batched code in Point class
    code = z_order_encode_(x, y, z, b=None, depth=depth)
    return code


def hilbert_encode(grid_coord: torch.Tensor, depth: int = 16):
    if depth == 0: depth = 1
    return hilbert_encode_(grid_coord, num_dims=3, num_bits=depth)

