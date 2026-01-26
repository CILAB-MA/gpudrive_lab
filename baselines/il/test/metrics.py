"""Metrics computation for WOSAC realism evaluation. Original code is from
https://github.com/Emerge-Lab/PufferDrive/blob/2.0/pufferlib/ocean/benchmark/metrics.py
https://github.com/Emerge-Lab/PufferDrive/blob/2.0/pufferlib/ocean/benchmark/interaction_features.py
https://github.com/Emerge-Lab/PufferDrive/blob/2.0/pufferlib/ocean/benchmark/map_metric_features.py
"""
import numpy as np
import torch
import math

EXTREMELY_LARGE_DISTANCE = 1e10
COLLISION_DISTANCE_THRESHOLD = 0.0
CORNER_ROUNDING_FACTOR = 0.7
MAX_HEADING_DIFF = math.radians(75.0)
MAX_HEADING_DIFF_FOR_SMALL_OVERLAP = math.radians(10.0)
SMALL_OVERLAP_THRESHOLD = 0.5
MAXIMUM_TIME_TO_COLLISION = 5.0

NUM_VERTICES_IN_BOX = 4

EXTREMELY_LARGE_DISTANCE = 1e10
OFFROAD_DISTANCE_THRESHOLD = 0.0

_METRIC_FIELD_NAMES = [
    "linear_speed",
    "linear_acceleration",
    "angular_speed",
    "angular_acceleration",
    "distance_to_nearest_object",
    "time_to_collision",
    "collision_indication",
    "distance_to_road_edge",
    "offroad_indication",
]

meta_data = dict(
    linear_speed=dict(
        min_val=0.0,
        max_val=25.0,
        num_bins=10,
        additive_smoothing=0.1,
        independent_timesteps=True,
        metametric_weight=0.05
    ),
    linear_acceleration=dict(
        min_val=-12.0,
        max_val=12.0,
        num_bins=11,
        additive_smoothing=0.1,
        independent_timesteps=True,
        metametric_weight=0.05
    ),
    angular_speed=dict(
        min_val=-0.628,
        max_val=0.628,
        num_bins=11,
        additive_smoothing=0.1,
        independent_timesteps=True,
        metametric_weight=0.05
    ),
    angular_acceleration=dict(
        min_val=-3.14,
        max_val=3.14,
        num_bins=11,
        additive_smoothing=0.1,
        independent_timesteps=True,
        metametric_weight=0.05
    ),
    distance_to_nearest_object=dict(
        min_val=-5.0,
        max_val=40.0,
        num_bins=10,
        additive_smoothing=0.1,
        independent_timesteps=True,
        metametric_weight=0.1
    ),
    time_to_collision=dict(
        min_val=0.0,
        max_val=5.0,
        num_bins=10,
        additive_smoothing=0.1,
        independent_timesteps=True,
        metametric_weight=0.1
    ),
    distance_to_road_edge=dict(
        min_val=-20.0,
        max_val=40.0,
        num_bins=10,
        additive_smoothing=0.1,
        independent_timesteps=True,
        metametric_weight=0.05
    ),
    collision_indication=dict(
        metametric_weight=0.25,
        bernoulli = True
    ),
    offroad_indication=dict(
        metametric_weight=0.25,
        bernoulli = True
    ),
)

def get_yaw_rotation_2d(heading: torch.Tensor) -> torch.Tensor:
    """Gets 2D rotation matrices from heading angles.

    Args:
        heading: Rotation angles in radians, any shape

    Returns:
        Rotation matrices, shape [..., 2, 2]
    """
    cos_heading = torch.cos(heading)
    sin_heading = torch.sin(heading)

    return torch.stack(
        [torch.stack([cos_heading, -sin_heading], dim=-1), torch.stack([sin_heading, cos_heading], dim=-1)], dim=-2
    )


def cross_product_2d(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Computes signed magnitude of cross product of 2D vectors.

    Args:
        a: Tensor with shape (..., 2)
        b: Tensor with same shape as a

    Returns:
        Cross product a[0]*b[1] - a[1]*b[0], shape (...)
    """
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


def _get_downmost_edge_in_box(box: torch.Tensor):
    """Finds the downmost (lowest y-coordinate) edge in the box.

    Assumes box edges are given in counter-clockwise order.

    Args:
        box: Tensor of shape (num_boxes, num_points_per_box, 2) with x-y coordinates

    Returns:
        Tuple of:
            - downmost_vertex_idx: Index of downmost vertex, shape (num_boxes, 1)
            - downmost_edge_direction: Tangent unit vector of downmost edge, shape (num_boxes, 1, 2)
    """
    downmost_vertex_idx = torch.argmin(box[..., 1], dim=-1).unsqueeze(-1)

    edge_start_vertex = torch.gather(box, 1, downmost_vertex_idx.unsqueeze(-1).expand(-1, -1, 2))
    edge_end_idx = torch.remainder(downmost_vertex_idx + 1, NUM_VERTICES_IN_BOX)
    edge_end_vertex = torch.gather(box, 1, edge_end_idx.unsqueeze(-1).expand(-1, -1, 2))

    downmost_edge = edge_end_vertex - edge_start_vertex
    downmost_edge_length = torch.linalg.norm(downmost_edge, dim=-1)
    downmost_edge_direction = downmost_edge / downmost_edge_length.unsqueeze(-1)

    return downmost_vertex_idx, downmost_edge_direction


def _get_edge_info(polygon_points: torch.Tensor):
    """Computes properties about the edges of a polygon.

    Args:
        polygon_points: Vertices of each polygon, shape (num_polygons, num_points_per_polygon, 2)

    Returns:
        Tuple of:
            - tangent_unit_vectors: Shape (num_polygons, num_points_per_polygon, 2)
            - normal_unit_vectors: Shape (num_polygons, num_points_per_polygon, 2)
            - edge_lengths: Shape (num_polygons, num_points_per_polygon)
    """
    first_point_in_polygon = polygon_points[:, 0:1, :]
    shifted_polygon_points = torch.cat([polygon_points[:, 1:, :], first_point_in_polygon], dim=-2)
    edge_vectors = shifted_polygon_points - polygon_points

    edge_lengths = torch.linalg.norm(edge_vectors, dim=-1)
    tangent_unit_vectors = edge_vectors / edge_lengths.unsqueeze(-1)
    normal_unit_vectors = torch.stack([-tangent_unit_vectors[..., 1], tangent_unit_vectors[..., 0]], dim=-1)

    return tangent_unit_vectors, normal_unit_vectors, edge_lengths


def get_2d_box_corners(boxes: torch.Tensor) -> torch.Tensor:
    """Given a set of 2D boxes, return its 4 corners.

    Args:
        boxes: Tensor of shape [..., 5] with [center_x, center_y, length, width, heading]

    Returns:
        Corners tensor of shape [..., 4, 2] in counter-clockwise order
    """
    center_x = boxes[..., 0]
    center_y = boxes[..., 1]
    length = boxes[..., 2]
    width = boxes[..., 3]
    heading = boxes[..., 4]

    rotation = get_yaw_rotation_2d(heading)
    translation = torch.stack([center_x, center_y], dim=-1)

    l2 = length * 0.5
    w2 = width * 0.5

    corners = torch.stack([l2, w2, -l2, w2, -l2, -w2, l2, -w2], dim=-1).reshape(boxes.shape[:-1] + (4, 2))

    corners = torch.einsum("...ij,...kj->...ki", rotation, corners) + translation.unsqueeze(-2)

    return corners


def minkowski_sum_of_box_and_box_points(box1_points: torch.Tensor, box2_points: torch.Tensor) -> torch.Tensor:
    """Batched Minkowski sum of two boxes (counter-clockwise corners in xy).

    Args:
        box1_points: Vertices for box 1, shape (num_boxes, 4, 2)
        box2_points: Vertices for box 2, shape (num_boxes, 4, 2)

    Returns:
        Minkowski sum of the two boxes, shape (num_boxes, 8, 2), in counter-clockwise order
    """
    device = box1_points.device
    point_order_1 = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3], dtype=torch.int64, device=device)
    point_order_2 = torch.tensor([0, 1, 1, 2, 2, 3, 3, 0], dtype=torch.int64, device=device)

    box1_start_idx, downmost_box1_edge_direction = _get_downmost_edge_in_box(box1_points)
    box2_start_idx, downmost_box2_edge_direction = _get_downmost_edge_in_box(box2_points)

    condition = cross_product_2d(downmost_box1_edge_direction, downmost_box2_edge_direction) >= 0.0
    condition = condition.repeat(1, 8)

    box1_point_order = torch.where(condition, point_order_2, point_order_1)
    box1_point_order = torch.remainder(box1_point_order + box1_start_idx, NUM_VERTICES_IN_BOX)
    ordered_box1_points = torch.gather(box1_points, 1, box1_point_order.unsqueeze(-1).expand(-1, -1, 2))

    box2_point_order = torch.where(condition, point_order_1, point_order_2)
    box2_point_order = torch.remainder(box2_point_order + box2_start_idx, NUM_VERTICES_IN_BOX)
    ordered_box2_points = torch.gather(box2_points, 1, box2_point_order.unsqueeze(-1).expand(-1, -1, 2))

    minkowski_sum = ordered_box1_points + ordered_box2_points

    return minkowski_sum


def dot_product_2d(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Computes the dot product of 2D vectors.

    Args:
        a: Tensor with shape (..., 2)
        b: Tensor with same shape as a

    Returns:
        Dot product a[0]*b[0] + a[1]*b[1], shape (...)
    """
    return a[..., 0] * b[..., 0] + a[..., 1] * b[..., 1]


def rotate_2d_points(xys: torch.Tensor, rotation_yaws: torch.Tensor) -> torch.Tensor:
    """Rotates xys counter-clockwise using rotation_yaws.

    Rotates about the origin counter-clockwise in the x-y plane.

    Args:
        xys: Tensor with shape (..., 2) containing xy coordinates
        rotation_yaws: Tensor with shape (...) containing angles in radians

    Returns:
        Rotated xys, shape (..., 2)
    """
    rel_cos_yaws = torch.cos(rotation_yaws)
    rel_sin_yaws = torch.sin(rotation_yaws)
    xs_out = rel_cos_yaws * xys[..., 0] - rel_sin_yaws * xys[..., 1]
    ys_out = rel_sin_yaws * xys[..., 0] + rel_cos_yaws * xys[..., 1]
    return torch.stack([xs_out, ys_out], dim=-1)


def signed_distance_from_point_to_convex_polygon(
    query_points: torch.Tensor, polygon_points: torch.Tensor
) -> torch.Tensor:
    """Finds signed distances from query points to convex polygons.

    Vertices must be ordered counter-clockwise.

    Args:
        query_points: Shape (batch_size, 2) with x-y coordinates
        polygon_points: Shape (batch_size, num_points_per_polygon, 2) with x-y coordinates

    Returns:
        Signed distances, shape (batch_size,). Negative if point is inside polygon.
    """
    tangent_unit_vectors, normal_unit_vectors, edge_lengths = _get_edge_info(polygon_points)

    query_points = query_points.unsqueeze(1)
    vertices_to_query_vectors = query_points - polygon_points
    vertices_distances = torch.linalg.norm(vertices_to_query_vectors, dim=-1)

    edge_signed_perp_distances = torch.sum(-normal_unit_vectors * vertices_to_query_vectors, dim=-1)

    is_inside = torch.all(edge_signed_perp_distances <= 0, dim=-1)

    projection_along_tangent = torch.sum(tangent_unit_vectors * vertices_to_query_vectors, dim=-1)
    projection_along_tangent_proportion = projection_along_tangent / edge_lengths

    is_projection_on_edge = (projection_along_tangent_proportion >= 0.0) & (projection_along_tangent_proportion <= 1.0)

    edge_perp_distances = torch.abs(edge_signed_perp_distances)
    edge_distances = torch.where(
        is_projection_on_edge, edge_perp_distances, torch.tensor(float("inf"), device=query_points.device)
    )

    edge_and_vertex_distance = torch.cat([edge_distances, vertices_distances], dim=-1)

    min_distance = torch.min(edge_and_vertex_distance, dim=-1).values
    signed_distances = torch.where(is_inside, -min_distance, min_distance)

    return signed_distances

def compute_signed_distances(
    center_x: torch.Tensor,
    center_y: torch.Tensor,
    length: torch.Tensor,
    width: torch.Tensor,
    heading: torch.Tensor,
    valid: torch.Tensor,
    evaluated_object_mask: torch.Tensor,
    corner_rounding_factor: float = CORNER_ROUNDING_FACTOR,
) -> torch.Tensor:
    """Computes pairwise signed distances between evaluated objects and all other objects.

    Objects are represented by 2D rectangles with rounded corners.

    Args:
        center_x: Shape (num_agents, num_rollouts, num_steps)
        center_y: Shape (num_agents, num_rollouts, num_steps)
        length: Shape (num_agents, num_rollouts) - constant per timestep
        width: Shape (num_agents, num_rollouts) - constant per timestep
        heading: Shape (num_agents, num_rollouts, num_steps)
        valid: Shape (num_agents, num_rollouts, num_steps)
        corner_rounding_factor: Rounding factor for box corners, between 0 (sharp) and 1 (capsule)

    Returns:
        signed_distances: shape (num_eval, num_agents, num_rollouts, num_steps)
    """

    num_agents = center_x.shape[0]
    num_rollouts = center_x.shape[1]
    num_steps = center_x.shape[2]

    eval_indices = torch.nonzero(evaluated_object_mask, as_tuple=False).squeeze(-1)
    num_eval = eval_indices.numel()

    if length.dim() == 2:
        length = length.unsqueeze(-1)
    if width.dim() == 2:
        width = width.unsqueeze(-1)
    length = length.expand(num_agents, num_rollouts, num_steps)
    width = width.expand(num_agents, num_rollouts, num_steps)

    boxes = torch.stack([center_x, center_y, length, width, heading], dim=-1)

    shrinking_distance = torch.minimum(boxes[..., 2], boxes[..., 3]) * corner_rounding_factor / 2.0

    shrunk_len = boxes[..., 2:3] - 2.0 * shrinking_distance.unsqueeze(-1)
    shrunk_wid = boxes[..., 3:4] - 2.0 * shrinking_distance.unsqueeze(-1)

    boxes = torch.cat(
        [
            boxes[..., :2],
            shrunk_len,
            shrunk_wid,
            boxes[..., 4:],
        ],
        dim=-1,
    )

    boxes_flat = boxes.reshape(num_agents * num_rollouts * num_steps, 5)
    box_corners = get_2d_box_corners(boxes_flat)
    box_corners = box_corners.reshape(num_agents, num_rollouts, num_steps, 4, 2)

    eval_corners = box_corners[eval_indices]

    batch_size = num_eval * num_agents * num_rollouts * num_steps

    corners_flat_1 = (
        eval_corners.unsqueeze(1).expand(num_eval, num_agents, num_rollouts, num_steps, 4, 2).reshape(batch_size, 4, 2)
    )

    corners_flat_2 = (
        box_corners.unsqueeze(0).expand(num_eval, num_agents, num_rollouts, num_steps, 4, 2).reshape(batch_size, 4, 2)
    )

    corners_flat_2.neg_()

    minkowski_sum = minkowski_sum_of_box_and_box_points(corners_flat_1, corners_flat_2)

    del corners_flat_1, corners_flat_2

    query_points = torch.zeros((batch_size, 2), dtype=center_x.dtype, device=center_x.device)

    signed_distances_flat = signed_distance_from_point_to_convex_polygon(
        query_points=query_points, polygon_points=minkowski_sum
    )

    del minkowski_sum, query_points

    signed_distances = signed_distances_flat.reshape(num_eval, num_agents, num_rollouts, num_steps)

    eval_shrinking = shrinking_distance[eval_indices]

    signed_distances.sub_(eval_shrinking[:, None, :, :])
    signed_distances.sub_(shrinking_distance[None, :, :, :])

    agent_indices = torch.arange(num_agents, device=center_x.device)
    self_mask = eval_indices[:, None] == agent_indices[None, :]

    self_mask = self_mask.unsqueeze(-1).unsqueeze(-1)

    signed_distances.masked_fill_(self_mask, EXTREMELY_LARGE_DISTANCE)

    eval_valid = valid[eval_indices]

    valid_mask = torch.logical_and(eval_valid[:, None, :, :], valid[None, :, :, :])

    signed_distances.masked_fill_(~valid_mask, EXTREMELY_LARGE_DISTANCE)

    return signed_distances


def compute_distance_to_nearest_object(
    center_x: torch.Tensor,
    center_y: torch.Tensor,
    length: torch.Tensor,
    width: torch.Tensor,
    heading: torch.Tensor,
    valid: torch.Tensor,
    evaluated_object_mask: torch.Tensor,
    corner_rounding_factor: float = CORNER_ROUNDING_FACTOR,
) -> torch.Tensor:
    signed_distances = compute_signed_distances(
        center_x=center_x,
        center_y=center_y,
        length=length,
        width=width,
        heading=heading,
        valid=valid,
        evaluated_object_mask=evaluated_object_mask,
        corner_rounding_factor=corner_rounding_factor,
    )

    min_distances = torch.min(signed_distances, dim=1).values
    return min_distances


def compute_time_to_collision(
    center_x: torch.Tensor,
    center_y: torch.Tensor,
    length: torch.Tensor,
    width: torch.Tensor,
    heading: torch.Tensor,
    valid: torch.Tensor,
    evaluated_object_mask: torch.Tensor,
    seconds_per_step: float,
) -> torch.Tensor:
    """Computes time-to-collision of the evaluated objects.

    The time-to-collision measures, in seconds, the time until an object collides
    with the object it is following, assuming constant speeds.

    Args:
        center_x: Shape (num_agents, num_rollouts, num_steps)
        center_y: Shape (num_agents, num_rollouts, num_steps)
        length: Shape (num_agents, num_rollouts) - constant per timestep
        width: Shape (num_agents, num_rollouts) - constant per timestep
        heading: Shape (num_agents, num_rollouts, num_steps)
        valid: Shape (num_agents, num_rollouts, num_steps)
        evaluated_object_mask: Shape (num_agents,) - boolean mask for evaluated agents
        seconds_per_step: Duration of one step in seconds

    Returns:
        Time-to-collision, shape (num_eval_agents, num_rollouts, num_steps)
    """

    valid = valid.to(dtype=torch.bool, device=center_x.device)
    evaluated_object_mask = evaluated_object_mask.to(dtype=torch.bool, device=center_x.device)

    num_agents = center_x.shape[0]
    num_rollouts = center_x.shape[1]
    num_steps = center_x.shape[2]

    eval_indices = torch.nonzero(evaluated_object_mask, as_tuple=False).squeeze(-1)
    num_eval = eval_indices.numel()

    # TODO: Convert to torch
    speed = compute_kinematic_features(
        x=center_x.cpu().numpy(),
        y=center_y.cpu().numpy(),
        heading=heading.cpu().numpy(),
        dt=seconds_per_step,
    )[0]
    if not isinstance(speed, torch.Tensor):
        speed = torch.as_tensor(speed, device=center_x.device, dtype=center_x.dtype)

    if length.dim() == 2:
        length = length.unsqueeze(-1)
    if width.dim() == 2:
        width = width.unsqueeze(-1)

    length = length.expand(num_agents, num_rollouts, num_steps).permute(2, 0, 1)
    width = width.expand(num_agents, num_rollouts, num_steps).permute(2, 0, 1)

    center_x = center_x.permute(2, 0, 1)
    center_y = center_y.permute(2, 0, 1)
    heading = heading.permute(2, 0, 1)
    speed = speed.permute(2, 0, 1)
    valid = valid.permute(2, 0, 1)

    ego_x = center_x[:, eval_indices]
    ego_y = center_y[:, eval_indices]
    ego_len = length[:, eval_indices]
    ego_wid = width[:, eval_indices]
    ego_heading = heading[:, eval_indices]
    ego_speed = speed[:, eval_indices]

    yaw_diff = torch.abs(heading.unsqueeze(1) - ego_heading.unsqueeze(2))

    yaw_diff_cos = torch.cos(yaw_diff)
    yaw_diff_sin = torch.sin(yaw_diff)

    all_sizes_half = torch.stack([length, width], dim=-1).unsqueeze(1) / 2.0

    other_long_offset = dot_product_2d(
        all_sizes_half,
        torch.abs(torch.stack([yaw_diff_cos, yaw_diff_sin], dim=-1)),
    )
    other_lat_offset = dot_product_2d(
        all_sizes_half,
        torch.abs(torch.stack([yaw_diff_sin, yaw_diff_cos], dim=-1)),
    )

    del all_sizes_half

    relative_x = center_x.unsqueeze(1) - ego_x.unsqueeze(2)
    relative_y = center_y.unsqueeze(1) - ego_y.unsqueeze(2)
    relative_xy = torch.stack([relative_x, relative_y], dim=-1)

    del relative_x, relative_y

    rotation = -ego_heading.unsqueeze(2).expand(-1, -1, num_agents, -1)

    other_relative_xy = rotate_2d_points(relative_xy, rotation)

    del relative_xy, rotation

    long_distance = other_relative_xy[..., 0] - ego_len.unsqueeze(2) / 2.0 - other_long_offset
    lat_overlap = torch.abs(other_relative_xy[..., 1]) - ego_wid.unsqueeze(2) / 2.0 - other_lat_offset

    del other_relative_xy, other_long_offset, other_lat_offset

    following_mask = _get_object_following_mask(
        long_distance.permute(1, 2, 0, 3),
        lat_overlap.permute(1, 2, 0, 3),
        yaw_diff.permute(1, 2, 0, 3),
    )

    del lat_overlap, yaw_diff

    valid_mask = torch.logical_and(valid.unsqueeze(1), following_mask.permute(2, 0, 1, 3))

    del following_mask

    long_distance.masked_fill_(~valid_mask, EXTREMELY_LARGE_DISTANCE)

    box_ahead_index = torch.argmin(long_distance, dim=2, keepdim=True)
    distance_to_box_ahead = torch.gather(long_distance, 2, box_ahead_index).squeeze(2)

    del long_distance

    speed_expanded = speed.unsqueeze(1).expand(-1, num_eval, -1, -1)
    box_ahead_speed = torch.gather(speed_expanded, 2, box_ahead_index).squeeze(2)

    rel_speed = ego_speed - box_ahead_speed

    rel_speed_safe = torch.where(rel_speed > 0.0, rel_speed, torch.ones_like(rel_speed))

    max_ttc = torch.full_like(rel_speed, MAXIMUM_TIME_TO_COLLISION)

    time_to_collision = torch.where(
        rel_speed > 0.0,
        torch.minimum(distance_to_box_ahead / rel_speed_safe, max_ttc),
        max_ttc,
    )

    return time_to_collision.permute(1, 2, 0)


def _get_object_following_mask(
    longitudinal_distance,
    lateral_overlap,
    yaw_diff,
):
    """Checks whether objects satisfy criteria for following another object.

    Args:
        longitudinal_distance: Shape (num_agents, num_agents, num_rollouts, num_steps)
            Longitudinal distances from back side of each ego box to other boxes.
        lateral_overlap: Shape (num_agents, num_agents, num_rollouts, num_steps)
            Lateral overlaps of other boxes over trails of ego boxes.
        yaw_diff: Shape (num_agents, num_agents, num_rollouts, num_steps)
            Absolute yaw differences between egos and other boxes.

    Returns:
        Boolean array indicating for each ego box if it is following the other boxes.
        Shape (num_agents, num_agents, num_rollouts, num_steps)
    """
    valid_mask = longitudinal_distance > 0.0
    valid_mask = torch.logical_and(valid_mask, yaw_diff <= MAX_HEADING_DIFF)
    valid_mask = torch.logical_and(valid_mask, lateral_overlap < 0.0)
    return torch.logical_and(
        valid_mask,
        torch.logical_or(
            lateral_overlap < -SMALL_OVERLAP_THRESHOLD,
            yaw_diff <= MAX_HEADING_DIFF_FOR_SMALL_OVERLAP,
        ),
    )

def _to_tensor(value, dtype, device=None):
    """Utility to convert numpy inputs to tensors on the requested device."""
    if isinstance(value, torch.Tensor):
        tensor = value
    else:
        tensor = torch.as_tensor(value, dtype=dtype)
    if dtype is not None and tensor.dtype != dtype:
        tensor = tensor.to(dtype)
    if device is not None and tensor.device != device:
        tensor = tensor.to(device)
    return tensor

def central_diff(t, pad_value):
    pad_shape = (*t.shape[:-1], 1)
    pad_array = np.full(pad_shape, pad_value)
    diff_t = (t[..., 2:] - t[..., :-2]) / 2
    return np.concatenate([pad_array, diff_t, pad_array], axis=-1)

def wrap_angle(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi

def compute_kinematic_features(x, y, heading, dt=0.1):
    dpos = central_diff(np.stack([x, y], axis=0), pad_value=np.nan)
    linear_speed = np.linalg.norm(dpos, ord=2, axis=0) / dt
    linear_accel = central_diff(linear_speed, pad_value=np.nan) / dt

    dh_step = wrap_angle(central_diff(heading, pad_value=np.nan) * 2) / 2
    dh = dh_step / dt
    d2h_step = wrap_angle(central_diff(dh_step, pad_value=np.nan) * 2) / 2
    d2h = d2h_step / (dt**2)

    return linear_speed, linear_accel, dh, d2h

def compute_kinematic_validity(valid):
    pad_shape = (*valid.shape[:-1], 1)
    pad_tensor = np.full(pad_shape, False)
    speed_validity = np.concatenate([pad_tensor, np.logical_and(valid[..., 2:], valid[..., :-2]), pad_tensor], axis=-1)

    pad_tensor = np.full(pad_shape, False)
    acceleration_validity = np.concatenate(
        [pad_tensor, np.logical_and(speed_validity[..., 2:], speed_validity[..., :-2]), pad_tensor], axis=-1
    )
    return speed_validity, acceleration_validity

def compute_interaction_features(
    xy: np.ndarray,
    heading: np.ndarray,
    scenario_ids: np.ndarray,
    agent_length: np.ndarray,
    agent_width: np.ndarray,
    eval_mask: np.ndarray,
    device: torch.device,
    valid: np.ndarray | None = None,
    corner_rounding_factor: float = 0.7,
    seconds_per_step: float = 0.1,
):
    x_t = _to_tensor(xy[:, :, 0, :], torch.float32, device=device)
    y_t = _to_tensor(xy[:, :, 1, :], torch.float32, device=device)
    heading_t = _to_tensor(heading, torch.float32, device=device)
    agent_length_t = _to_tensor(agent_length, torch.float32, device=device)
    agent_width_t = _to_tensor(agent_width, torch.float32, device=device)

    num_agents = x_t.shape[0]
    num_eval_agents = int(np.sum(eval_mask))
    num_rollouts = x_t.shape[1]
    num_steps = x_t.shape[2]

    if valid is None:
        valid_t = torch.ones((num_agents, num_rollouts, num_steps), dtype=torch.bool, device=x_t.device)
    else:
        valid_t = _to_tensor(valid, torch.bool, device=x_t.device)

    length_broadcast = agent_length_t.unsqueeze(-1).expand(num_agents, num_rollouts)
    width_broadcast = agent_width_t.unsqueeze(-1).expand(num_agents, num_rollouts)

    result_distances = np.full(
        (num_eval_agents, num_rollouts, num_steps), EXTREMELY_LARGE_DISTANCE, dtype=np.float32
    )
    result_collisions = np.full((num_eval_agents, num_rollouts, num_steps), False, dtype=bool)
    result_ttc = np.full(
        (num_eval_agents, num_rollouts, num_steps), MAXIMUM_TIME_TO_COLLISION, dtype=np.float32
    )

    unique_scenarios = np.unique(scenario_ids)

    eval_indices = np.where(eval_mask)[0]
    eval_to_result = {idx: i for i, idx in enumerate(eval_indices)}

    for scenario_id in unique_scenarios:
        scenario_mask_np = scenario_ids[:, 0] == scenario_id
        agent_indices = np.where(scenario_mask_np)[0]
        if agent_indices.size == 0:
            continue
        scenario_mask = torch.as_tensor(scenario_mask_np, dtype=torch.bool, device=x_t.device)
        scenario_x = x_t[scenario_mask]
        scenario_y = y_t[scenario_mask]
        episode_length = length_broadcast[scenario_mask]
        scenario_width = width_broadcast[scenario_mask]
        scenario_heading = heading_t[scenario_mask]
        scenario_valid = valid_t[scenario_mask]

        scenario_eval_mask_np = eval_mask[scenario_mask_np]
        scenario_eval_mask = torch.as_tensor(scenario_eval_mask_np, dtype=torch.bool, device=x_t.device)

        distances_to_objects = compute_distance_to_nearest_object(
            center_x=scenario_x,
            center_y=scenario_y,
            length=episode_length,
            width=scenario_width,
            heading=scenario_heading,
            valid=scenario_valid,
            corner_rounding_factor=corner_rounding_factor,
            evaluated_object_mask=scenario_eval_mask,
        )

        is_colliding_per_step = distances_to_objects < COLLISION_DISTANCE_THRESHOLD

        times_to_collision = compute_time_to_collision(
            center_x=scenario_x,
            center_y=scenario_y,
            length=episode_length,
            width=scenario_width,
            heading=scenario_heading,
            valid=scenario_valid,
            seconds_per_step=seconds_per_step,
            evaluated_object_mask=scenario_eval_mask,
        )

        eval_agents_in_scenario = agent_indices[scenario_eval_mask_np]
        result_indices = [eval_to_result[idx] for idx in eval_agents_in_scenario]

        distances_np = distances_to_objects.cpu().numpy()
        collisions_np = is_colliding_per_step.cpu().numpy()
        ttc_np = times_to_collision.cpu().numpy()

        result_distances[result_indices] = distances_np
        result_collisions[result_indices] = collisions_np
        result_ttc[result_indices] = ttc_np

    return result_distances, result_collisions, result_ttc

def compute_distance_to_road_edge(
    center_x: torch.Tensor,
    center_y: torch.Tensor,
    length: torch.Tensor,
    width: torch.Tensor,
    heading: torch.Tensor,
    valid: torch.Tensor,
    polyline_x: torch.Tensor,
    polyline_y: torch.Tensor,
    polyline_lengths: torch.Tensor,
) -> torch.Tensor:
    num_agents, num_steps = center_x.shape

    if length.ndim == 1:
        length = length.unsqueeze(-1).expand(-1, num_steps)
    if width.ndim == 1:
        width = width.unsqueeze(-1).expand(-1, num_steps)

    boxes = torch.stack([center_x, center_y, length, width, heading], dim=-1)
    boxes_flat = boxes.reshape(-1, 5)

    corners = get_2d_box_corners(boxes_flat)
    corners = corners.reshape(num_agents, num_steps, 4, 2)

    flat_corners = corners.reshape(-1, 2)

    polylines_padded, polylines_valid = _pad_polylines(polyline_x, polyline_y, polyline_lengths)

    corner_distances = _compute_signed_distance_to_polylines(flat_corners, polylines_padded, polylines_valid)

    corner_distances = corner_distances.reshape(num_agents, num_steps, 4)
    signed_distances = torch.max(corner_distances, dim=-1).values

    offroad_fill = signed_distances.new_full((), -EXTREMELY_LARGE_DISTANCE)
    signed_distances = torch.where(valid, signed_distances, offroad_fill)

    return signed_distances


def _pad_polylines(
    polyline_x: torch.Tensor,
    polyline_y: torch.Tensor,
    polyline_lengths: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    device = polyline_x.device
    num_polylines = polyline_lengths.shape[0]
    max_length = int(polyline_lengths.max().item())

    polylines = torch.zeros((num_polylines, max_length, 2), dtype=torch.float32, device=device)
    valid = torch.zeros((num_polylines, max_length), dtype=torch.bool, device=device)

    lengths_long = polyline_lengths.to(torch.long)
    boundaries = torch.cumsum(torch.cat([lengths_long.new_zeros(1), lengths_long]), dim=0)

    for i in range(num_polylines):
        start = int(boundaries[i].item())
        end = int(boundaries[i + 1].item())
        length_i = int(lengths_long[i].item())
        polylines[i, :length_i, 0] = polyline_x[start:end]
        polylines[i, :length_i, 1] = polyline_y[start:end]
        valid[i, :length_i] = True

    return polylines, valid


def _check_polyline_cycles(
    polylines: torch.Tensor,
    polylines_valid: torch.Tensor,
    tolerance: float = 1e-3,
) -> torch.Tensor:
    device = polylines.device
    max_length = polylines.shape[1]
    valid_counts = polylines_valid.sum(dim=-1)
    has_enough_points = valid_counts >= 2

    indices = torch.arange(max_length, device=device)
    last_idx = torch.argmax(polylines_valid.int() * indices, dim=-1)

    first_pts = polylines[:, 0]
    gather_idx = last_idx.view(-1, 1, 1).expand(-1, 1, 2)
    last_pts = torch.gather(polylines, 1, gather_idx).squeeze(1)
    dist = torch.linalg.norm(first_pts - last_pts, dim=-1)

    return (dist < tolerance) & has_enough_points


def _compute_signed_distance_to_polylines(
    xys: torch.Tensor,
    polylines: torch.Tensor,
    polylines_valid: torch.Tensor,
) -> torch.Tensor:
    num_points = xys.shape[0]
    num_polylines, max_length = polylines.shape[:2]
    num_segments = max_length - 1

    is_segment_valid = polylines_valid[:, :-1] & polylines_valid[:, 1:]
    is_polyline_cyclic = _check_polyline_cycles(polylines, polylines_valid)

    xy_starts = polylines[:, :-1, :]
    xy_ends = polylines[:, 1:, :]
    start_to_end = xy_ends - xy_starts

    start_to_point = xys.unsqueeze(0).unsqueeze(0) - xy_starts[:, :, None, :]

    dot_se_se = dot_product_2d(start_to_end, start_to_end)
    dot_sp_se = dot_product_2d(start_to_point, start_to_end[:, :, None, :])

    denom = dot_se_se[:, :, None]
    rel_t = torch.where(
        denom != 0,
        dot_sp_se / denom,
        torch.zeros_like(dot_sp_se),
    )

    n = torch.sign(cross_product_2d(start_to_point, start_to_end[:, :, None, :]))

    segment_to_point = start_to_point - (start_to_end[:, :, None, :] * torch.clamp(rel_t, 0.0, 1.0)[:, :, :, None])
    distance_to_segment_2d = torch.linalg.norm(segment_to_point, dim=-1)

    start_to_end_padded = torch.cat(
        [
            start_to_end[:, -1:, :],
            start_to_end,
            start_to_end[:, :1, :],
        ],
        dim=1,
    )

    is_locally_convex = (
        cross_product_2d(start_to_end_padded[:, :-1, None, :], start_to_end_padded[:, 1:, None, :]) > 0.0
    )

    n_prior = torch.cat(
        [
            torch.where(
                is_polyline_cyclic[:, None, None],
                n[:, -1:, :],
                n[:, :1, :],
            ),
            n[:, :-1, :],
        ],
        dim=1,
    )
    n_next = torch.cat(
        [
            n[:, 1:, :],
            torch.where(
                is_polyline_cyclic[:, None, None],
                n[:, :1, :],
                n[:, -1:, :],
            ),
        ],
        dim=1,
    )

    is_prior_valid = torch.cat(
        [
            torch.where(
                is_polyline_cyclic[:, None],
                is_segment_valid[:, -1:],
                is_segment_valid[:, :1],
            ),
            is_segment_valid[:, :-1],
        ],
        dim=1,
    )
    is_next_valid = torch.cat(
        [
            is_segment_valid[:, 1:],
            torch.where(
                is_polyline_cyclic[:, None],
                is_segment_valid[:, :1],
                is_segment_valid[:, -1:],
            ),
        ],
        dim=1,
    )

    sign_if_before = torch.where(
        is_locally_convex[:, :-1, :],
        torch.maximum(n, n_prior),
        torch.minimum(n, n_prior),
    )
    sign_if_after = torch.where(
        is_locally_convex[:, 1:, :],
        torch.maximum(n, n_next),
        torch.minimum(n, n_next),
    )

    sign_to_segment = torch.where(
        (rel_t < 0.0) & is_prior_valid[:, :, None],
        sign_if_before,
        torch.where((rel_t > 1.0) & is_next_valid[:, :, None], sign_if_after, n),
    )

    distance_to_segment_2d = distance_to_segment_2d.reshape(num_polylines * num_segments, num_points).T
    sign_to_segment = sign_to_segment.reshape(num_polylines * num_segments, num_points).T

    is_segment_valid_flat = is_segment_valid.reshape(num_polylines * num_segments)
    valid_mask = is_segment_valid_flat.unsqueeze(0).expand(num_points, -1)
    distance_to_segment_2d = distance_to_segment_2d.masked_fill(
        ~valid_mask,
        EXTREMELY_LARGE_DISTANCE,
    )

    closest_idx = torch.argmin(distance_to_segment_2d, dim=1)
    point_indices = torch.arange(num_points, device=xys.device)
    distance_2d = distance_to_segment_2d[point_indices, closest_idx]
    distance_sign = sign_to_segment[point_indices, closest_idx]

    return distance_sign * distance_2d


def compute_map_features(
    xy: np.ndarray,
    heading: np.ndarray,
    scenario_ids: np.ndarray,
    agent_length: np.ndarray,
    agent_width: np.ndarray,
    road_edge_polylines: dict,
    device: torch.device,
    valid: np.ndarray | None = None,
):
    x_t = _to_tensor(xy[:, :, 0, :], torch.float32, device=device)
    y_t = _to_tensor(xy[:, :, 1, :], torch.float32, device=device)
    heading_t = _to_tensor(heading, torch.float32, device=device)
    agent_length_t = _to_tensor(agent_length, torch.float32, device=device)
    agent_width_t = _to_tensor(agent_width, torch.float32, device=device)
    num_agents = x_t.shape[0]
    num_rollouts = x_t.shape[1]
    num_steps = x_t.shape[2]

    if valid is None:
        valid_t = torch.ones((num_agents, num_rollouts, num_steps), dtype=torch.bool, device=device)
    else:
        valid_t = _to_tensor(valid, torch.bool, device=device)

    result_distances = np.zeros((num_agents, num_rollouts, num_steps), dtype=np.float32)
    result_offroad = np.zeros((num_agents, num_rollouts, num_steps), dtype=bool)

    unique_scenarios = np.unique(scenario_ids)

    polyline_boundaries = np.cumsum(np.concatenate([[0], road_edge_polylines["lengths"]]))

    for scenario_id in unique_scenarios:
        agent_mask_np = scenario_ids[:, 0] == scenario_id
        agent_indices = np.where(agent_mask_np)[0]

        if len(agent_indices) == 0:
            continue

        polyline_mask = road_edge_polylines["scenario_id"] == scenario_id
        polyline_indices = np.where(polyline_mask)[0]

        scenario_lengths = road_edge_polylines["lengths"][polyline_mask]

        scenario_x_list = []
        scenario_y_list = []
        for idx in polyline_indices:
            start = polyline_boundaries[idx]
            end = polyline_boundaries[idx + 1]
            scenario_x_list.append(road_edge_polylines["x"][start:end])
            scenario_y_list.append(road_edge_polylines["y"][start:end])

        scenario_polyline_x = torch.as_tensor(np.concatenate(scenario_x_list), dtype=torch.float32, device=x_t.device)
        scenario_polyline_y = torch.as_tensor(np.concatenate(scenario_y_list), dtype=torch.float32, device=x_t.device)
        scenario_lengths_t = torch.as_tensor(scenario_lengths, dtype=torch.int64, device=x_t.device)

        agent_mask = torch.as_tensor(agent_mask_np, dtype=torch.bool, device=x_t.device)
        scenario_x = x_t[agent_mask]
        scenario_y = y_t[agent_mask]
        scenario_heading = heading_t[agent_mask]
        scenario_valid = valid_t[agent_mask]
        scenario_length = agent_length_t[agent_mask]
        scenario_width = agent_width_t[agent_mask]

        for rollout_idx in range(num_rollouts):
            distances = compute_distance_to_road_edge(
                center_x=scenario_x[:, rollout_idx, :],
                center_y=scenario_y[:, rollout_idx, :],
                length=scenario_length,
                width=scenario_width,
                heading=scenario_heading[:, rollout_idx, :],
                valid=scenario_valid[:, rollout_idx, :],
                polyline_x=scenario_polyline_x,
                polyline_y=scenario_polyline_y,
                polyline_lengths=scenario_lengths_t,
            )

            distances_np = distances.cpu().numpy()
            result_distances[agent_mask_np, rollout_idx, :] = distances_np
            result_offroad[agent_mask_np, rollout_idx, :] = (
                distances_np > OFFROAD_DISTANCE_THRESHOLD
            )

    return result_distances, result_offroad

def compute_displacement_error(
    pred_x: np.ndarray,
    pred_y: np.ndarray,
    ref_x: np.ndarray,
    ref_y: np.ndarray,
    ref_valid: np.ndarray,
) -> np.ndarray:

    # Compute displacement error for each timestep and every agent and rollout
    ref_traj = np.stack([ref_x, ref_y], axis=-1)  # (n_agents, 1, n_steps, 2)
    pred_traj = np.stack([pred_x, pred_y], axis=-1)
    displacement = np.linalg.norm(pred_traj - ref_traj, axis=-1)  # (n_agents, n_rollouts, n_steps)

    # Mask invalid timesteps
    displacement = np.where(ref_valid, displacement, 0.0)

    # Aggregate
    valid_count = np.sum(ref_valid, axis=2)  # (n_agents, 1)

    # Compute ADE
    ade_per_rollout = np.sum(displacement, axis=2) / np.maximum(valid_count, 1)  # (n_agents, n_rollouts)

    ade = ade_per_rollout.mean(axis=-1)  # (n_agents,)

    # The rollout with the minimum ADE for each agent
    min_ade = np.min(ade_per_rollout, axis=1)  # (n_agents,)

    return ade, min_ade

def log_likelihood_estimate_timeseries(
    log_values: np.ndarray,
    sim_values: np.ndarray,
    meta_data: dict,
    # min_val: float,
    # max_val: float,
    # num_bins: int,
    # additive_smoothing: float,
    # treat_timesteps_independently: bool = True,
    # sanity_check: bool = False,
    # plot_agent_idx: int = 0,
) -> np.ndarray:
    n_agents, n_rollouts, n_steps = sim_values.shape
    min_val = meta_data["min_val"]
    max_val = meta_data["max_val"]
    num_bins = meta_data["num_bins"]
    additive_smoothing = meta_data["additive_smoothing"]
    treat_timesteps_independently = meta_data["independent_timesteps"]
    if treat_timesteps_independently:
        # Ignore temporal structure: We end up with (n_agents, n_rollouts * n_steps)
        log_flat = log_values.reshape(n_agents, n_steps)
        sim_flat = sim_values.reshape(n_agents, n_rollouts * n_steps)

    else:
        # If values in time are instead to be compared per-step, reshape:
        # - `sim_values` as (n_objects * n_steps, n_rollouts)
        # - `log_values` as (n_objects * n_steps, 1)
        log_flat = log_values.reshape(n_agents * n_steps, 1)
        sim_flat = sim_values.transpose(0, 2, 1).reshape(n_agents * n_steps, n_rollouts)
    
    # Compute log-likelihoods
    log_probs = histogram_estimate(log_flat, sim_flat, min_val, max_val, num_bins, additive_smoothing)

    # Depending on `independent_timesteps`, the likelihoods might be flattened, so
    # reshape back to the initial `log_values` shape.
    log_probs = log_probs.reshape(n_agents, n_steps)

    # Sanity check visualization
    # if sanity_check:
    #     _plot_histogram_sanity_check(log_flat, sim_flat, log_probs, plot_agent_idx)

    return log_probs

def histogram_estimate(
    log_samples: np.ndarray,
    sim_samples: np.ndarray,
    min_val: float,
    max_val: float,
    num_bins: int,
    additive_smoothing: float,
) -> np.ndarray:

    n_agents, sample_size = sim_samples.shape

    # Clip samples to valid range
    log_samples_clipped = np.clip(log_samples, min_val, max_val)
    sim_samples_clipped = np.clip(sim_samples, min_val, max_val)

    # Create bin edges
    edges = np.linspace(min_val, max_val, num_bins + 1)

    # Create histogram for each agent from sim samples
    sim_counts = np.array([np.histogram(sim_samples_clipped[i], bins=edges)[0] for i in range(n_agents)])

    # Apply smoothing and normalize to probabilities
    sim_counts = sim_counts.astype(float) + additive_smoothing
    sim_probs = sim_counts / sim_counts.sum(axis=1, keepdims=True)

    # Find which bin each log sample belongs to
    # digitize returns values in [1, num_bins], so subtract 1 for 0-indexing
    # right=False means bins are [left, right) except last bin which is [left, right]
    log_bins = np.digitize(log_samples_clipped, edges, right=False) - 1

    # Clip to valid bin indices (handles edge case where value == max_val)
    log_bins = np.clip(log_bins, 0, num_bins - 1)

    # Get log probabilities for each sample
    agent_indices = np.arange(n_agents)[:, None]
    log_probs = np.log(sim_probs[agent_indices, log_bins])

    return log_probs

def reduce_average_with_validity(tensor: np.ndarray, validity: np.ndarray, axis: int = None) -> np.ndarray:
    if tensor.shape != validity.shape:
        raise ValueError(
            f"Shapes of `tensor` and `validity` must be the same. (Actual: {tensor.shape}, {validity.shape})."
        )
    cond_sum = np.sum(np.where(validity, tensor, np.zeros_like(tensor)), axis=axis, keepdims=False)
    valid_sum = np.sum(validity.astype(np.float32), axis=axis, keepdims=False)

    # Safe division:
    safe_valid_sum = np.where(valid_sum == 0, 1, valid_sum)

    return np.where(valid_sum == 0, np.nan, cond_sum / safe_valid_sum)

def log_likelihood_estimate_scenario_level(
    log_values: np.ndarray,
    sim_values: np.ndarray,
    min_val: float,
    max_val: float,
    num_bins: int,
    additive_smoothing: float | None = None,
    use_bernoulli: bool = False,
) -> np.ndarray:
    if log_values.ndim != 1:
        raise ValueError(f"log_values must be 1D, got shape {log_values.shape}")
    if sim_values.ndim != 2:
        raise ValueError(f"sim_values must be 2D, got shape {sim_values.shape}")

    log_values_2d = log_values[:, np.newaxis]
    sim_values_2d = sim_values

    if use_bernoulli:
        log_likelihood_2d = bernoulli_estimate(
            log_values_2d.astype(bool),
            sim_values_2d.astype(bool),
            additive_smoothing=0.001,
        )
    else:
        log_likelihood_2d = histogram_estimate(
            log_values_2d,
            sim_values_2d,
            min_val=min_val,
            max_val=max_val,
            num_bins=num_bins,
            additive_smoothing=additive_smoothing,
        )

    return log_likelihood_2d[:, 0]

def bernoulli_estimate(
    log_samples: np.ndarray,
    sim_samples: np.ndarray,
    additive_smoothing: float,
) -> np.ndarray:
    if log_samples.dtype != bool:
        raise ValueError("log_samples must be boolean array for Bernoulli estimate")
    if sim_samples.dtype != bool:
        raise ValueError("sim_samples must be boolean array for Bernoulli estimate")

    return histogram_estimate(
        log_samples.astype(float),
        sim_samples.astype(float),
        min_val=-0.5,
        max_val=1.5,
        num_bins=2,
        additive_smoothing=additive_smoothing,
    )

def compute_metametric(metrics) -> float:
        metametric = 0.0
        for field_name in _METRIC_FIELD_NAMES:
            likelihood_field_name = "likelihood_" + field_name
            weight = meta_data[field_name]["metametric_weight"]
            metric_score = metrics[likelihood_field_name]
            metametric += weight * metric_score
        return metametric