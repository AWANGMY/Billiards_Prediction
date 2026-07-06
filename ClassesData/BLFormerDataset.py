import math

import torch
from torch.utils.data import Dataset


TABLE_WIDTH = 200.0
TABLE_HEIGHT = 100.0
TABLE_DIAGONAL = math.sqrt(TABLE_WIDTH * TABLE_WIDTH + TABLE_HEIGHT * TABLE_HEIGHT)
BALL_RADIUS = 57.15 / 2540.0 * TABLE_WIDTH / 2.0
PAIR_FEATURE_DIM = 10

TOKEN_CLS = 0
TOKEN_BALL = 1
TOKEN_POCKET = 2
TOKEN_PAD = 3

EDGE_CLS_OR_PAD = 0
EDGE_BALL_BALL = 1
EDGE_BALL_POCKET = 2
EDGE_POCKET_POCKET = 3

POCKETS = torch.tensor([[0.0, 0.0],
                        [0.0, TABLE_HEIGHT],
                        [TABLE_WIDTH / 2.0, TABLE_HEIGHT],
                        [TABLE_WIDTH, TABLE_HEIGHT],
                        [TABLE_WIDTH, 0.0],
                        [TABLE_WIDTH / 2.0, 0.0]], dtype=torch.float32)


def angle_between(vector_1, vector_2):

    norm_1 = torch.linalg.norm(vector_1)
    norm_2 = torch.linalg.norm(vector_2)

    if float(norm_1.item()) == 0.0 or float(norm_2.item()) == 0.0:
        return 0.0

    cosine = torch.dot(vector_1, vector_2) / (norm_1 * norm_2)
    cosine = torch.clamp(cosine, -1.0, 1.0)

    return float(torch.rad2deg(torch.acos(cosine)).item())


def angle_between_xy(x1, y1, x2, y2):

    norm_1 = math.sqrt(x1 * x1 + y1 * y1)
    norm_2 = math.sqrt(x2 * x2 + y2 * y2)

    if norm_1 == 0.0 or norm_2 == 0.0:
        return 0.0

    cosine = (x1 * x2 + y1 * y2) / (norm_1 * norm_2)
    cosine = max(-1.0, min(1.0, cosine))

    return math.degrees(math.acos(cosine))


def pocket_cushion_angle(ball_position, pocket_position):

    pocket_x = float(pocket_position[0].item())
    pocket_y = float(pocket_position[1].item())

    if pocket_x == 0.0 and pocket_y == 0.0:
        directions = [(1.0, 0.0), (0.0, 1.0)]
    elif pocket_x == 0.0 and pocket_y == TABLE_HEIGHT:
        directions = [(1.0, 0.0), (0.0, -1.0)]
    elif pocket_x == TABLE_WIDTH and pocket_y == TABLE_HEIGHT:
        directions = [(-1.0, 0.0), (0.0, -1.0)]
    elif pocket_x == TABLE_WIDTH and pocket_y == 0.0:
        directions = [(-1.0, 0.0), (0.0, 1.0)]
    else:
        directions = [(1.0, 0.0), (-1.0, 0.0)]

    line_vector = ball_position - pocket_position
    angles = []
    for direction in directions:
        direction_tensor = torch.tensor(direction, dtype=ball_position.dtype,
                                        device=ball_position.device)
        angles.append(angle_between(line_vector, direction_tensor))

    return min(angles)


def pocket_cushion_angle_xy(ball_position, pocket_position):

    ball_x, ball_y = ball_position
    pocket_x, pocket_y = pocket_position

    if pocket_x == 0.0 and pocket_y == 0.0:
        directions = [(1.0, 0.0), (0.0, 1.0)]
    elif pocket_x == 0.0 and pocket_y == TABLE_HEIGHT:
        directions = [(1.0, 0.0), (0.0, -1.0)]
    elif pocket_x == TABLE_WIDTH and pocket_y == TABLE_HEIGHT:
        directions = [(-1.0, 0.0), (0.0, -1.0)]
    elif pocket_x == TABLE_WIDTH and pocket_y == 0.0:
        directions = [(-1.0, 0.0), (0.0, 1.0)]
    else:
        directions = [(1.0, 0.0), (-1.0, 0.0)]

    line_x = ball_x - pocket_x
    line_y = ball_y - pocket_y

    return min([angle_between_xy(line_x, line_y, direction[0], direction[1])
                for direction in directions])


def point_segment_distance_xy(point, start, end):

    point_x, point_y = point
    start_x, start_y = start
    end_x, end_y = end

    segment_x = end_x - start_x
    segment_y = end_y - start_y
    segment_length_squared = segment_x * segment_x + segment_y * segment_y

    if segment_length_squared == 0.0:
        return math.hypot(point_x - start_x, point_y - start_y)

    t = ((point_x - start_x) * segment_x + (point_y - start_y) * segment_y)
    t = t / segment_length_squared
    t = max(0.0, min(1.0, t))
    projection_x = start_x + t * segment_x
    projection_y = start_y + t * segment_y

    return math.hypot(point_x - projection_x, point_y - projection_y)


def path_blocked_xy(start, end, active_ball_positions, excluded_ball_indices):

    start_x, start_y = start
    end_x, end_y = end
    segment_x = end_x - start_x
    segment_y = end_y - start_y
    segment_length_squared = segment_x * segment_x + segment_y * segment_y

    if segment_length_squared == 0.0:
        return 0.0

    for ball_index, ball_position in enumerate(active_ball_positions):
        if ball_index in excluded_ball_indices:
            continue

        ball_x, ball_y = ball_position
        t = ((ball_x - start_x) * segment_x + (ball_y - start_y) * segment_y)
        t = t / segment_length_squared
        if t <= 0.0 or t >= 1.0:
            continue

        distance = point_segment_distance_xy(ball_position, start, end)
        if distance < 2.0 * BALL_RADIUS:
            return 1.0

    return 0.0


def min_ball_to_pocket_angle_xy(incoming_start, object_ball):

    start_x, start_y = incoming_start
    object_x, object_y = object_ball
    incoming_x = object_x - start_x
    incoming_y = object_y - start_y
    best_angle = 180.0

    for pocket_x, pocket_y in POCKETS.tolist():
        outgoing_x = pocket_x - object_x
        outgoing_y = pocket_y - object_y
        best_angle = min(best_angle,
                         angle_between_xy(incoming_x, incoming_y,
                                          outgoing_x, outgoing_y))

    return best_angle


def point_segment_distance(point, start, end):

    segment = end - start
    segment_length_squared = torch.dot(segment, segment)

    if float(segment_length_squared.item()) == 0.0:
        return float(torch.linalg.norm(point - start).item())

    t = torch.dot(point - start, segment) / segment_length_squared
    t = torch.clamp(t, 0.0, 1.0)
    projection = start + t * segment

    return float(torch.linalg.norm(point - projection).item())


def path_blocked(start, end, active_ball_positions, excluded_ball_indices):

    segment = end - start
    if float(torch.linalg.norm(segment).item()) == 0.0:
        return 0.0

    for ball_index, ball_position in enumerate(active_ball_positions):
        if ball_index in excluded_ball_indices:
            continue

        segment_length_squared = torch.dot(segment, segment)
        t = torch.dot(ball_position - start, segment) / segment_length_squared
        if float(t.item()) <= 0.0 or float(t.item()) >= 1.0:
            continue

        distance = point_segment_distance(ball_position, start, end)
        if distance < 2.0 * BALL_RADIUS:
            return 1.0

    return 0.0


def min_ball_to_pocket_angle(incoming_start, object_ball):

    incoming_vector = object_ball - incoming_start
    best_angle = 180.0

    for pocket_position in POCKETS:
        outgoing_vector = pocket_position - object_ball
        best_angle = min(best_angle, angle_between(incoming_vector, outgoing_vector))

    return best_angle


def edge_type(token_type_i, token_type_j):

    if token_type_i in [TOKEN_CLS, TOKEN_PAD] or token_type_j in [TOKEN_CLS, TOKEN_PAD]:
        return EDGE_CLS_OR_PAD
    if token_type_i == TOKEN_BALL and token_type_j == TOKEN_BALL:
        return EDGE_BALL_BALL
    if token_type_i == TOKEN_POCKET and token_type_j == TOKEN_POCKET:
        return EDGE_POCKET_POCKET

    return EDGE_BALL_POCKET


def build_pair_features(token_type_ids, physical_coords, ball_token_positions):

    token_type_list = token_type_ids.tolist()
    coord_list = physical_coords.tolist()
    ball_token_list = ball_token_positions.tolist()
    length = len(token_type_list)
    pair_features = [[[0.0] * PAIR_FEATURE_DIM for _ in range(length)]
                     for _ in range(length)]

    active_ball_positions = [coord_list[token_index] for token_index in ball_token_list]
    token_to_active_ball = {}
    for active_index, token_index in enumerate(ball_token_list):
        token_to_active_ball[int(token_index)] = active_index

    for i in range(length):
        type_i = int(token_type_list[i])
        for j in range(length):
            type_j = int(token_type_list[j])
            current_edge_type = edge_type(type_i, type_j)
            pair_features[i][j][6 + current_edge_type] = 1.0

            if current_edge_type == EDGE_CLS_OR_PAD:
                continue

            start = coord_list[i]
            end = coord_list[j]
            displacement_x = end[0] - start[0]
            displacement_y = end[1] - start[1]
            distance = math.hypot(displacement_x, displacement_y)

            pair_features[i][j][0] = distance / TABLE_DIAGONAL
            pair_features[i][j][1] = displacement_x / TABLE_WIDTH
            pair_features[i][j][2] = displacement_y / TABLE_HEIGHT

            if current_edge_type == EDGE_BALL_BALL:
                pair_features[i][j][3] = min_ball_to_pocket_angle_xy(start, end) / 180.0
                excluded = {token_to_active_ball[i], token_to_active_ball[j]}
                pair_features[i][j][4] = path_blocked_xy(start, end, active_ball_positions,
                                                          excluded)
            elif current_edge_type == EDGE_BALL_POCKET:
                if type_i == TOKEN_BALL:
                    ball_position = start
                    pocket_position = end
                    excluded = {token_to_active_ball[i]}
                else:
                    ball_position = end
                    pocket_position = start
                    excluded = {token_to_active_ball[j]}

                pair_features[i][j][3] = (
                    pocket_cushion_angle_xy(ball_position, pocket_position) / 90.0)
                pair_features[i][j][4] = path_blocked_xy(ball_position, pocket_position,
                                                          active_ball_positions,
                                                          excluded)

    return torch.tensor(pair_features, dtype=torch.float32)


class BLFormerDataset(Dataset):

    def __init__(self, data, indices, augment=False):

        super(BLFormerDataset, self).__init__()

        self.data = data
        self.indices = indices.long() if isinstance(indices, torch.Tensor) else torch.tensor(indices, dtype=torch.long)
        self.augment = augment

        for target in ['clear', 'win', 'potted_after_break']:
            if target not in self.data:
                raise ValueError('Missing target: ' + target)

    def __len__(self):

        return len(self.indices)

    def __getitem__(self, idx):

        data_idx = int(self.indices[idx].item())
        layout = self.data['x'][data_idx].clone().float()

        if self.augment:
            layout = self._augment_layout(layout)

        paper_layout = None
        if 'x_paper' in self.data:
            paper_layout = self.data['x_paper'][data_idx].clone().long()

        sample = self._layout_to_tokens(layout, paper_layout)
        sample['clear'] = self.data['clear'][data_idx].clone().long()
        sample['win'] = self.data['win'][data_idx].clone().long()
        sample['potted_after_break'] = self.data['potted_after_break'][data_idx].clone().long()
        sample['sample_index'] = torch.tensor(data_idx, dtype=torch.long)

        return sample

    def _augment_layout(self, layout):

        if torch.rand(1).item() < 0.5:
            layout[:, 0] = 1.0 - layout[:, 0]
        if torch.rand(1).item() < 0.5:
            layout[:, 1] = 1.0 - layout[:, 1]

        return layout

    def _layout_to_tokens(self, layout, paper_layout=None):

        token_type_ids = [TOKEN_CLS]
        coords = [[0.0, 0.0]]
        ball_ids = [0]
        paper_features = [[0] * 27]
        ball_token_positions = []

        for ball_index in range(layout.shape[0]):
            if float(layout[ball_index, 2].item()) <= 0.5:
                continue

            ball_token_positions.append(len(token_type_ids))
            token_type_ids.append(TOKEN_BALL)
            coords.append([float(layout[ball_index, 0].item()),
                           float(layout[ball_index, 1].item())])
            ball_ids.append(ball_index + 1)
            if paper_layout is None:
                paper_features.append([0] * 27)
            else:
                paper_features.append(paper_layout[ball_index].tolist())

        for pocket_position in POCKETS:
            token_type_ids.append(TOKEN_POCKET)
            coords.append([float((pocket_position[0] / TABLE_WIDTH).item()),
                           float((pocket_position[1] / TABLE_HEIGHT).item())])
            ball_ids.append(0)
            paper_features.append([0] * 27)

        token_type_ids = torch.tensor(token_type_ids, dtype=torch.long)
        coords = torch.tensor(coords, dtype=torch.float32)
        ball_ids = torch.tensor(ball_ids, dtype=torch.long)
        paper_features = torch.tensor(paper_features, dtype=torch.long)
        physical_coords = coords.clone()
        physical_coords[:, 0] = physical_coords[:, 0] * TABLE_WIDTH
        physical_coords[:, 1] = physical_coords[:, 1] * TABLE_HEIGHT
        ball_token_positions = torch.tensor(ball_token_positions, dtype=torch.long)
        pair_features = build_pair_features(token_type_ids, physical_coords,
                                            ball_token_positions)

        return {'token_type_ids': token_type_ids,
                'coords': coords,
                'ball_ids': ball_ids,
                'paper_features': paper_features,
                'attention_mask': torch.ones(len(token_type_ids), dtype=torch.bool),
                'pair_features': pair_features}


def blformer_collate(samples):

    batch_size = len(samples)
    max_length = max([int(sample['token_type_ids'].numel()) for sample in samples])

    token_type_ids = torch.full((batch_size, max_length), TOKEN_PAD, dtype=torch.long)
    coords = torch.zeros(batch_size, max_length, 2, dtype=torch.float32)
    ball_ids = torch.zeros(batch_size, max_length, dtype=torch.long)
    paper_features = torch.zeros(batch_size, max_length, 27, dtype=torch.long)
    attention_mask = torch.zeros(batch_size, max_length, dtype=torch.bool)
    pair_features = torch.zeros(batch_size, max_length, max_length,
                                PAIR_FEATURE_DIM, dtype=torch.float32)
    pair_features[:, :, :, 6 + EDGE_CLS_OR_PAD] = 1.0

    clear = torch.zeros(batch_size, dtype=torch.long)
    win = torch.zeros(batch_size, dtype=torch.long)
    potted_after_break = torch.zeros(batch_size, dtype=torch.long)
    sample_indices = torch.zeros(batch_size, dtype=torch.long)

    for batch_index, sample in enumerate(samples):
        length = int(sample['token_type_ids'].numel())
        token_type_ids[batch_index, :length] = sample['token_type_ids']
        coords[batch_index, :length] = sample['coords']
        ball_ids[batch_index, :length] = sample['ball_ids']
        paper_features[batch_index, :length] = sample['paper_features']
        attention_mask[batch_index, :length] = sample['attention_mask']
        pair_features[batch_index, :length, :length] = sample['pair_features']
        clear[batch_index] = sample['clear']
        win[batch_index] = sample['win']
        potted_after_break[batch_index] = sample['potted_after_break']
        sample_indices[batch_index] = sample['sample_index']

    return {'token_type_ids': token_type_ids,
            'coords': coords,
            'ball_ids': ball_ids,
            'paper_features': paper_features,
            'attention_mask': attention_mask,
            'pair_features': pair_features,
            'clear': clear,
            'win': win,
            'potted_after_break': potted_after_break,
            'sample_indices': sample_indices}
