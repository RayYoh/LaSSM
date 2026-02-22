import torch
import fps_ops_cuda


def point_sampler(fps_type, xyz, npoints, features=None, scores=None, weight_gamma=1.0):
    """
    Args: 
        fps_type: in ['d-fps', 'f-fps', 's-fps']
        xyz: (B, N, 3)
        features: (B, C, N)
        scores: (B, N)
    """
    if fps_type == 'd-fps':
        sample_idx = furthest_point_sample(xyz.contiguous(), npoints)
    elif fps_type == 'f-fps':
        dist_matrix = calc_dist_matrix_for_sampling(xyz.contiguous(), features.permute(0, 2, 1), weight_gamma)
        sample_idx = furthest_point_sample_matrix(dist_matrix, npoints)
    elif fps_type == 's-fps':
        scores = scores.sigmoid() ** weight_gamma
        sample_idx = furthest_point_sample_weights(xyz.contiguous(), scores.contiguous(), npoints)
    return sample_idx


@torch.no_grad()
def calc_dist_matrix_for_sampling(xyz: torch.Tensor, features: torch.Tensor = None,
                                  gamma: float = 1.0):
    dist = torch.cdist(xyz, xyz)
    
    if features is not None:
        dist += torch.cdist(features, features) * gamma
    
    return dist


@torch.no_grad()
def furthest_point_sample(xyz: torch.Tensor, npoint: int) -> torch.Tensor:
    """
    Uses iterative furthest point sampling to select a set of npoint features that have the largest
    minimum distance
    :param ctx:
    :param xyz: (B, N, 3) where N > npoint
    :param npoint: int, number of features in the sampled set
    :return:
            output: (B, npoint) tensor containing the set
    """
    assert xyz.is_contiguous()

    B, N, _ = xyz.size()
    output = torch.cuda.IntTensor(B, npoint)
    temp = torch.cuda.FloatTensor(B, N).fill_(1e10)

    fps_ops_cuda.furthest_point_sampling_wrapper(B, N, npoint, xyz, temp, output)
    return output


@torch.no_grad()
def furthest_point_sample_matrix(matrix: torch.Tensor, npoint: int) -> torch.Tensor:
    """
    Uses iterative furthest point sampling to select a set of npoint features that have the largest
    minimum distance with a pairwise distance matrix
    :param matrix: (B, N, N) tensor of dist matrix
    :param npoint: int, number of features in the sampled set
    :return:
         output: (B, npoint) tensor containing the set
    """
    assert matrix.is_contiguous()

    B, N, _ = matrix.size()
    output = torch.cuda.IntTensor(B, npoint)
    temp = torch.cuda.FloatTensor(B, N).fill_(1e10)

    fps_ops_cuda.furthest_point_sampling_matrix_wrapper(B, N, npoint, matrix, temp, output)
    return output


@torch.no_grad()
def furthest_point_sample_weights(xyz: torch.Tensor, weights: torch.Tensor, npoint: int) -> torch.Tensor:
    """
    Uses iterative furthest point sampling to select a set of npoint features that have the largest
    minimum weighted distance
    Args:
        xyz: (B, N, 3), tensor of xyz coordinates
        weights: (B, N), tensor of point weights
        npoint: int, number of points in the sampled set
    Returns:
        output: (B, npoint) tensor containing the set
    """
    assert xyz.is_contiguous()
    assert weights.is_contiguous()

    B, N, _ = xyz.size()
    output = torch.cuda.IntTensor(B, npoint)
    temp = torch.cuda.FloatTensor(B, N).fill_(1e10)

    fps_ops_cuda.furthest_point_sampling_weights_wrapper(B, N, npoint, xyz, weights, temp, output)
    return output
