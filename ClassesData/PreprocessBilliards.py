import ast
import json
import math
import os
import re
import zipfile
from collections import Counter
import xml.etree.ElementTree as ET

import numpy as np
import torch


class BilliardsPreprocessor:

    def __init__(self, root='Dataset', output_path=None):

        self.root = root
        self.layout_root = os.path.join(self.root, 'data_layouts')
        self.coordinates_root = os.path.join(self.layout_root, 'All cordinates')
        self.variables_root = os.path.join(self.layout_root, 'Variables')

        if output_path is None:
            output_path = os.path.join(self.root, 'processed', 'billiards_layout.pt')

        self.output_path = output_path
        self.last_audit = None

    def preprocess(self,
                   train_ratio=0.7,
                   val_ratio=0.1,
                   seed=123,
                   drop_bad_remarks=True,
                   deduplicate=True,
                   clamp_coordinates=True,
                   split_method='paper',
                   audit_json=None,
                   save=True):

        samples = self._build_samples(drop_bad_remarks=drop_bad_remarks,
                                      deduplicate=deduplicate,
                                      clamp_coordinates=clamp_coordinates)

        if split_method == 'paper':
            split_indices = self._split_by_paper(samples,
                                                 test_ratio=1.0 - train_ratio,
                                                 val_ratio=val_ratio,
                                                 seed=seed)
        elif split_method == 'paper40':
            split_indices = self._split_by_paper40(samples,
                                                   train_ratio=0.4,
                                                   seed=seed)
        elif split_method == 'group':
            split_indices = self._split_by_group(samples,
                                                 train_ratio=train_ratio,
                                                 val_ratio=val_ratio,
                                                 seed=seed)
        else:
            raise ValueError('Unknown split method: ' + str(split_method))

        data = self._pack_samples(samples, split_indices, split_method)
        self._finalize_audit(data, split_method, seed)

        if save:
            output_dir = os.path.dirname(self.output_path)
            if output_dir != '':
                os.makedirs(output_dir, exist_ok=True)
            torch.save(data, self.output_path)

        if audit_json is not None:
            self._write_audit_json(audit_json)

        return data

    def _build_samples(self,
                       drop_bad_remarks=True,
                       deduplicate=True,
                       clamp_coordinates=True):

        xml_index = self._index_coordinate_xml_files()
        variables_files = self._list_files(self.variables_root, '.xlsx')
        coordinate_xml_files = self._list_files(self.coordinates_root, '.xml')

        audit = {'settings': {'drop_bad_remarks': drop_bad_remarks,
                              'deduplicate': deduplicate,
                              'clamp_coordinates': clamp_coordinates},
                 'input': {'variables_files': len(variables_files),
                           'coordinate_xml_files': len(coordinate_xml_files),
                           'indexed_coordinate_xml_files': len(xml_index)},
                 'filters': Counter(),
                 'label_distribution': {},
                 'accepted_samples': 0,
                 'unique_groups': 0}

        samples = []
        seen_layouts = set()

        for variables_file in variables_files:
            rows = self._read_variables_rows(variables_file)
            audit['filters']['raw_rows'] += len(rows)

            for row in rows:
                frame = self._to_int(self._get_value(row, ['frame']))
                if frame is None:
                    audit['filters']['missing_frame'] += 1
                    continue

                potted_after_break = self._to_int(self._get_value(row, ['potted after break']))
                potted_when_break = self._to_int(self._get_value(row, ['potted when break']))
                clear = self._to_int(self._get_value(row, ['clear']))
                win = self._to_int(self._get_value(row, ['win']))

                if not self._valid_count_label(potted_after_break):
                    audit['filters']['invalid_potted_after_break'] += 1
                    continue
                if clear not in [0, 1] or win not in [0, 1]:
                    audit['filters']['invalid_clear_or_win'] += 1
                    continue

                remarks = str(self._get_value(row, ['remarks']) or '').lower()
                if drop_bad_remarks and self._is_bad_remark(remarks):
                    audit['filters']['bad_remarks'] += 1
                    continue

                xml_file = self._find_coordinate_xml(variables_file, frame, xml_index)
                if xml_file is None:
                    audit['filters']['missing_coordinate_xml'] += 1
                    continue

                positions = self._read_layout_xml(xml_file)
                if positions is None or len(positions) != 10:
                    audit['filters']['invalid_or_non_10_ball_xml'] += 1
                    continue

                positions = self._normalize_positions(positions,
                                                      clamp_coordinates=clamp_coordinates)
                exists = np.ones(10, dtype=np.float32)

                for ball_id in self._get_potted_on_break(row, potted_when_break):
                    if ball_id is not None and 1 <= ball_id <= 9:
                        exists[ball_id] = 0.0
                        positions[ball_id] = np.array([0.0, 0.0], dtype=np.float32)

                ball_ids = np.arange(10, dtype=np.float32) / 9.0
                features = np.concatenate([positions,
                                           exists[:, None],
                                           ball_ids[:, None]], axis=1)
                paper_features = self._paper_token_features(positions, exists, cell=15)

                layout_key = tuple(np.round(features.reshape(-1), 4).tolist())
                if deduplicate and layout_key in seen_layouts:
                    audit['filters']['duplicate_layout'] += 1
                    continue
                seen_layouts.add(layout_key)

                group_id = self._get_group_id(variables_file)

                samples.append({'features': features,
                                'paper_features': paper_features,
                                'potted_after_break': potted_after_break,
                                'potted_when_break': potted_when_break if potted_when_break is not None else 0,
                                'clear': clear,
                                'win': win,
                                'sample_id': group_id + '_frame_' + str(frame),
                                'group_id': group_id,
                                'frame': frame,
                                'xml_file': xml_file,
                                'variables_file': variables_file})

        audit['accepted_samples'] = len(samples)
        audit['unique_groups'] = len(set([sample['group_id'] for sample in samples]))
        for target in ['clear', 'win', 'potted_after_break', 'potted_when_break']:
            values = [sample[target] for sample in samples]
            audit['label_distribution'][target] = dict(sorted(Counter(values).items()))
        audit['filters'] = dict(sorted(audit['filters'].items()))
        self.last_audit = audit

        if len(samples) == 0:
            raise RuntimeError('No valid billiards samples were found under ' + self.layout_root)

        return samples

    def _pack_samples(self, samples, split_indices, split_method):

        x = torch.tensor(np.stack([sample['features'] for sample in samples]), dtype=torch.float32)
        x_paper = torch.tensor(np.stack([sample['paper_features'] for sample in samples]), dtype=torch.long)
        potted_after_break = torch.tensor([sample['potted_after_break'] for sample in samples], dtype=torch.long)
        potted_when_break = torch.tensor([sample['potted_when_break'] for sample in samples], dtype=torch.long)
        clear = torch.tensor([sample['clear'] for sample in samples], dtype=torch.long)
        win = torch.tensor([sample['win'] for sample in samples], dtype=torch.long)

        return {'x': x,
                'x_paper': x_paper,
                'potted_after_break': potted_after_break,
                'potted_when_break': potted_when_break,
                'clear': clear,
                'win': win,
                'sample_ids': [sample['sample_id'] for sample in samples],
                'group_ids': [sample['group_id'] for sample in samples],
                'frames': [sample['frame'] for sample in samples],
                'xml_files': [sample['xml_file'] for sample in samples],
                'variables_files': [sample['variables_file'] for sample in samples],
                'split_indices': split_indices,
                'split_method': split_method,
                'feature_names': ['x_norm', 'y_norm', 'exists', 'ball_id_norm'],
                'paper_feature_names': ['position',
                                        'pocket_0_angle', 'pocket_0_distance', 'pocket_0_clear_path', 'pocket_0',
                                        'pocket_1_angle', 'pocket_1_distance', 'pocket_1_clear_path', 'pocket_1',
                                        'pocket_2_angle', 'pocket_2_distance', 'pocket_2_clear_path', 'pocket_2',
                                        'pocket_3_angle', 'pocket_3_distance', 'pocket_3_clear_path', 'pocket_3',
                                        'pocket_4_angle', 'pocket_4_distance', 'pocket_4_clear_path', 'pocket_4',
                                        'pocket_5_angle', 'pocket_5_distance', 'pocket_5_clear_path', 'pocket_5',
                                        'best_angle', 'best_pocket'],
                'paper_cell': 15,
                'input_dim': tuple(x.shape[1:]),
                'paper_input_dim': tuple(x_paper.shape[1:]),
                'n_classes': {'potted_after_break': 10,
                              'potted_when_break': 10,
                              'clear': 2,
                              'win': 2}}

    def _split_by_group(self, samples, train_ratio=0.7, val_ratio=0.15, seed=123):

        groups = sorted(list(set([sample['group_id'] for sample in samples])))

        rng = np.random.default_rng(seed)
        rng.shuffle(groups)

        train_end = int(len(groups) * train_ratio)
        val_end = int(len(groups) * (train_ratio + val_ratio))

        train_groups = set(groups[:train_end])
        val_groups = set(groups[train_end:val_end])
        test_groups = set(groups[val_end:])

        train_indices = []
        val_indices = []
        test_indices = []

        for index, sample in enumerate(samples):
            if sample['group_id'] in train_groups:
                train_indices.append(index)
            elif sample['group_id'] in val_groups:
                val_indices.append(index)
            elif sample['group_id'] in test_groups:
                test_indices.append(index)

        return {'train': torch.tensor(train_indices, dtype=torch.long),
                'val': torch.tensor(val_indices, dtype=torch.long),
                'test': torch.tensor(test_indices, dtype=torch.long)}

    def _split_by_paper(self, samples, test_ratio=0.3, val_ratio=0.1, seed=123):

        num_data = len(samples)
        indices_data = list(range(num_data))
        split_tt = int(np.floor(test_ratio * num_data))

        train_indices = indices_data[split_tt:]
        test_indices = indices_data[:split_tt]

        rng = np.random.default_rng(seed)
        shuffled_train = list(range(len(train_indices)))
        rng.shuffle(shuffled_train)

        split_tv = int(np.floor(val_ratio * len(train_indices)))
        val_indices = [train_indices[index] for index in shuffled_train[:split_tv]]
        train_new_indices = [train_indices[index] for index in shuffled_train[split_tv:]]

        return {'train': torch.tensor(train_new_indices, dtype=torch.long),
                'val': torch.tensor(val_indices, dtype=torch.long),
                'test': torch.tensor(test_indices, dtype=torch.long)}

    def _split_by_paper40(self, samples, train_ratio=0.4, seed=123):

        num_data = len(samples)
        indices_data = np.arange(num_data)
        rng = np.random.default_rng(seed)
        rng.shuffle(indices_data)

        split = int(np.floor(train_ratio * num_data))
        train_indices = indices_data[:split].tolist()
        test_indices = indices_data[split:].tolist()

        return {'train': torch.tensor(train_indices, dtype=torch.long),
                'val': torch.tensor([], dtype=torch.long),
                'test': torch.tensor(test_indices, dtype=torch.long)}

    def _finalize_audit(self, data, split_method, seed):

        if self.last_audit is None:
            return

        self.last_audit['split'] = {'method': split_method,
                                    'seed': seed,
                                    'sizes': {name: len(indices)
                                              for name, indices in data['split_indices'].items()}}
        self.last_audit['paper_feature_stats'] = self._paper_feature_stats(
            data['x_paper'], data['paper_feature_names'])

    def _paper_feature_stats(self, x_paper, feature_names):

        stats = []
        for index, name in enumerate(feature_names):
            values = x_paper[:, :, index].reshape(-1)
            stats.append({'index': index,
                          'name': name,
                          'min': int(values.min().item()),
                          'max': int(values.max().item()),
                          'unique': int(torch.unique(values).numel())})

        return stats

    def _write_audit_json(self, audit_json):

        if self.last_audit is None:
            return

        output_dir = os.path.dirname(audit_json)
        if output_dir != '':
            os.makedirs(output_dir, exist_ok=True)

        with open(audit_json, 'w', encoding='utf-8') as output_file:
            json.dump(self.last_audit, output_file, indent=2, sort_keys=True)

    def _read_variables_rows(self, variables_file):

        raw_rows = self._read_xlsx_rows(variables_file)

        header_index = None
        for index, row in enumerate(raw_rows[:20]):
            values = [str(value).strip().lower() for value in row if value not in [None, '']]
            if len(values) > 0 and values[0] == 'frame':
                header_index = index
                break

        if header_index is None:
            return []

        headers = []
        for index, value in enumerate(raw_rows[header_index]):
            if value in [None, '']:
                headers.append('unnamed_' + str(index))
            else:
                headers.append(str(value).strip().lower())

        rows = []
        for raw_row in raw_rows[header_index + 1:]:
            row = {}
            for index, header in enumerate(headers):
                row[header] = raw_row[index] if index < len(raw_row) else None
            rows.append(row)

        return rows

    def _read_xlsx_rows(self, file_path):

        try:
            workbook = zipfile.ZipFile(file_path)
        except zipfile.BadZipFile:
            return []

        with workbook:
            shared_strings = self._read_shared_strings(workbook)
            sheet_file = self._get_first_sheet_file(workbook)
            root = ET.fromstring(workbook.read(sheet_file))

        namespace = {'main': 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
        rows = []

        for row in root.findall('.//main:sheetData/main:row', namespace):
            values = []

            for cell in row.findall('main:c', namespace):
                cell_ref = cell.attrib.get('r')
                if cell_ref is not None:
                    column_index = self._column_index(cell_ref)
                    while len(values) <= column_index:
                        values.append(None)

                value = self._read_xlsx_cell(cell, shared_strings, namespace)

                if cell_ref is None:
                    values.append(value)
                else:
                    values[column_index] = value

            rows.append(values)

        return rows

    def _read_shared_strings(self, workbook):

        if 'xl/sharedStrings.xml' not in workbook.namelist():
            return []

        root = ET.fromstring(workbook.read('xl/sharedStrings.xml'))
        namespace = {'main': 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
        shared_strings = []

        for item in root.findall('main:si', namespace):
            text_parts = []
            for text in item.findall('.//main:t', namespace):
                text_parts.append(text.text or '')
            shared_strings.append(''.join(text_parts))

        return shared_strings

    def _get_first_sheet_file(self, workbook):

        sheet_files = [name for name in workbook.namelist()
                       if name.startswith('xl/worksheets/sheet') and name.endswith('.xml')]
        sheet_files = sorted(sheet_files)

        if len(sheet_files) == 0:
            raise RuntimeError('No worksheet found in xlsx file')

        return sheet_files[0]

    def _read_xlsx_cell(self, cell, shared_strings, namespace):

        cell_type = cell.attrib.get('t')

        if cell_type == 'inlineStr':
            text = cell.find('.//main:t', namespace)
            return '' if text is None or text.text is None else text.text

        value = cell.find('main:v', namespace)
        if value is None or value.text is None:
            return None

        if cell_type == 's':
            return shared_strings[int(value.text)]
        if cell_type == 'b':
            return int(value.text)

        return self._to_number(value.text)

    def _read_layout_xml(self, xml_file):

        namespace = {'ss': 'urn:schemas-microsoft-com:office:spreadsheet'}

        try:
            root = ET.parse(xml_file).getroot()
        except ET.ParseError:
            return None

        positions = []

        for row in root.findall('.//ss:Row', namespace):
            values = []
            for data in row.findall('.//ss:Data', namespace):
                values.append(data.text if data.text is not None else '')

            if len(values) >= 3 and str(values[0]).startswith('Marker'):
                x = self._to_float(values[1])
                y = self._to_float(values[2])
                if x is None or y is None:
                    return None
                positions.append([x, y])

        return positions

    def _normalize_positions(self, positions, clamp_coordinates=True):

        positions = np.array(positions, dtype=np.float32)
        max_x = np.nanmax(positions[:, 0])
        max_y = np.nanmax(positions[:, 1])
        x_span = np.nanmax(positions[:, 0]) - np.nanmin(positions[:, 0])
        y_span = np.nanmax(positions[:, 1]) - np.nanmin(positions[:, 1])

        if max_x > 120 or (max_y <= 120 and x_span >= y_span):
            normalized = np.stack([positions[:, 0] / 200.0,
                                   positions[:, 1] / 100.0], axis=1)
        else:
            normalized = np.stack([positions[:, 1] / 200.0,
                                   positions[:, 0] / 100.0], axis=1)

        if clamp_coordinates:
            normalized = np.clip(normalized, 0.0, 1.0)

        return normalized.astype(np.float32)


    def _paper_token_features(self, positions, exists, cell=15):

        coordinates = self._paper_coordinates(positions)
        active_indices = [index for index in range(10) if exists[index] > 0.5]

        paper_features = []
        for ball_id in range(10):
            if exists[ball_id] <= 0.5:
                paper_features.append([0] * 27)
                continue

            active_position = active_indices.index(ball_id)
            later_indices = active_indices[active_position + 1:]
            later_balls = [coordinates[index] for index in later_indices]

            if active_position > 0:
                previous_ball = coordinates[active_indices[active_position - 1]]
            else:
                previous_ball = None

            ball_feature = self._paper_ball_feature(coordinates[ball_id],
                                                    later_balls,
                                                    previous_ball,
                                                    ball_id == 0,
                                                    cell)
            paper_features.append(ball_feature)

        return np.array(paper_features, dtype=np.int64)

    def _paper_coordinates(self, positions):

        x = positions[:, 1] * 100.0
        y = positions[:, 0] * 200.0
        coordinates = np.stack([x, y], axis=1)
        coordinates[:, 0] = np.clip(coordinates[:, 0], 0.0, 99.0)
        coordinates[:, 1] = np.clip(coordinates[:, 1], 0.0, 199.0)

        return coordinates.astype(np.float32)

    def _paper_ball_feature(self, point, later_balls, previous_ball, is_cue_ball, cell):

        pockets = [(0, 0), (100, 0), (0, 100), (0, 200), (100, 100), (100, 200)]
        angle_settings = self._paper_angle_settings(point, is_cue_ball, cell)

        ball_feature = [self._paper_position_token(point, cell)]
        possible_pockets = []

        for pocket_index, pocket in enumerate(pockets):
            angle_id = self._paper_angle_token(angle_settings[pocket_index][0],
                                               angle_settings[pocket_index][1],
                                               angle_settings[pocket_index][2])
            distance_id = self._paper_distance_token(point, pocket)

            if is_cue_ball:
                clear_path = 0
            else:
                clear_path = self._paper_clear_path(point, pocket, later_balls)
                if clear_path == 1:
                    possible_pockets.append(pocket)

            ball_feature.extend([angle_id, distance_id, clear_path, pocket_index])

        if is_cue_ball or len(possible_pockets) == 0 or previous_ball is None:
            ball_feature.extend([0, 0])
        else:
            best_angle = None
            best_pocket = possible_pockets[0]
            incoming_vector = (point[0] - previous_ball[0], point[1] - previous_ball[1])

            for pocket in possible_pockets:
                outgoing_vector = (pocket[0] - point[0], pocket[1] - point[1])
                angle = math.acos(round(self._cosine_distance(incoming_vector, outgoing_vector), 2)) * 180 / math.pi
                angle_id = self._clip_id(int(angle / 5), 37)

                if best_angle is None or angle_id < best_angle:
                    best_angle = angle_id
                    best_pocket = pocket

            ball_feature.extend([best_angle, pockets.index(best_pocket)])

        return ball_feature

    def _paper_angle_settings(self, point, is_cue_ball, cell):

        x = float(point[0])
        y = float(point[1])

        if is_cue_ball:
            x_cell = int(x / cell)
            y_cell = int(y / cell)

            return [((0, 1), (x, y), 90),
                    ((-100, 0), (x - 100, y), 90),
                    ((0, 100), (x, y - 100), 180),
                    ((0, -200), (x, y - 200), 90),
                    ((0, 100), (x - 100, y - 100), 180),
                    ((0, -100), (x_cell - 100, y_cell - 200), 90)]

        return [((0, 1), (x, y), 90),
                ((100, 0), (100 - x, 0 - y), 90),
                ((0, 100), (0 - x, 100 - y), 180),
                ((0, 200), (0 - x, 200 - y), 90),
                ((0, 100), (100 - x, 100 - y), 180),
                ((0, -100), (x - 100, y - 200), 90)]

    def _paper_position_token(self, point, cell):

        x_token = int(float(point[0]) / cell)
        y_token = int(float(point[1]) / cell)
        token = y_token * math.ceil(100 / cell) + x_token

        return self._clip_id(token, 200)

    def _paper_angle_token(self, vector_1, vector_2, max_angle):

        angle = math.acos(self._cosine_distance(vector_2, vector_1)) * 180 / math.pi
        angle = min(angle, max_angle - angle)
        angle = max(0.0, angle)

        if max_angle == 90:
            return self._clip_id(int(angle / 15), 4)

        return self._clip_id(int(angle / 15), 7)

    def _paper_distance_token(self, point, pocket):

        distance = self._cal_distance(point, pocket)

        return self._clip_id(int(distance / 10), 25)

    def _paper_clear_path(self, point, pocket, ball_list):

        x_min = min(float(point[0]), pocket[0])
        y_min = min(float(point[1]), pocket[1])
        x_max = max(float(point[0]), pocket[0])
        y_max = max(float(point[1]), pocket[1])

        for ball in ball_list:
            if float(ball[0]) >= x_min and float(ball[0]) <= x_max and float(ball[1]) >= y_min and float(ball[1]) <= y_max:
                distance = self._paper_point_line_distance(point, ball, pocket)
                if distance <= 3.0:
                    return 0

        return 1

    def _paper_point_line_distance(self, start, middle, end):

        a = end[1] - start[1]
        b = start[0] - end[0]
        c = end[0] * start[1] - start[0] * end[1]

        if a == 0 and b == 0:
            return 0.0

        return abs((a * middle[0] + b * middle[1] + c) / math.sqrt(a * a + b * b))

    def _cal_distance(self, p1, p2):

        return math.sqrt(math.pow(float(p2[0]) - float(p1[0]), 2) + math.pow(float(p2[1]) - float(p1[1]), 2))

    def _cosine_distance(self, p1, p2):

        denominator = math.sqrt(p1[0] * p1[0] + p1[1] * p1[1]) * math.sqrt(p2[0] * p2[0] + p2[1] * p2[1])

        if denominator == 0:
            return 0.0

        value = (p1[0] * p2[0] + p1[1] * p2[1]) / denominator

        return max(-1.0, min(1.0, value))

    def _clip_id(self, value, size):

        return max(0, min(int(value), size - 1))

    def _index_coordinate_xml_files(self):

        xml_files = self._list_files(self.coordinates_root, '.xml')
        xml_index = {}

        for xml_file in xml_files:
            frame = self._frame_from_xml_name(xml_file)
            if frame is None:
                continue

            rel_dir = os.path.relpath(os.path.dirname(xml_file), self.coordinates_root)
            parts = rel_dir.split(os.sep)

            if len(parts) < 2:
                continue

            event_name = self._clean_name(parts[0])
            match_name = self._clean_name(parts[-1])
            xml_index[(event_name, match_name, frame)] = xml_file

        return xml_index

    def _find_coordinate_xml(self, variables_file, frame, xml_index):

        rel_dir = os.path.relpath(os.path.dirname(variables_file), self.variables_root)
        exact_dir = os.path.join(self.coordinates_root, rel_dir)

        candidates = [os.path.join(exact_dir, 'Frame ' + str(frame) + '.xml'),
                      os.path.join(exact_dir, str(frame) + '.xml')]

        for candidate in candidates:
            if os.path.exists(candidate):
                return candidate

        parts = rel_dir.split(os.sep)
        if len(parts) < 2:
            return None

        event_name = self._clean_name(parts[0])
        match_name = self._clean_name(parts[-1])

        return xml_index.get((event_name, match_name, frame))

    def _get_potted_on_break(self, row, potted_when_break):

        if potted_when_break is None or potted_when_break <= 0:
            return []

        order_value = self._get_value(row, ['order'])
        if order_value is None or order_value == '':
            return []

        try:
            order = ast.literal_eval(str(order_value))
        except (SyntaxError, ValueError):
            return []

        if not isinstance(order, list) or len(order) == 0:
            return []

        first_item = order[0]
        if isinstance(first_item, tuple) or isinstance(first_item, list):
            return [self._to_int(item) for item in first_item]

        potted_balls = []
        for item in order:
            if isinstance(item, tuple) or isinstance(item, list):
                for sub_item in item:
                    potted_balls.append(self._to_int(sub_item))
            else:
                potted_balls.append(self._to_int(item))

            if len(potted_balls) >= potted_when_break:
                break

        return potted_balls[:potted_when_break]

    def _get_group_id(self, variables_file):

        rel_dir = os.path.relpath(os.path.dirname(variables_file), self.variables_root)
        return rel_dir.replace(os.sep, '__')

    def _get_value(self, row, names):

        for name in names:
            name = name.lower()
            for key, value in row.items():
                if key.lower() == name:
                    return value

        return None

    def _is_bad_remark(self, remarks):

        bad_words = ['cue ball', 'cueball', 'illegal', 'foul', 'golden break']

        for word in bad_words:
            if word in remarks:
                return True

        return False

    def _valid_count_label(self, value):

        return value is not None and 0 <= value <= 9

    def _list_files(self, root, suffix):

        files = []

        for current_root, _, file_names in os.walk(root):
            for file_name in file_names:
                if file_name.startswith('~$'):
                    continue
                if file_name.lower().endswith(suffix):
                    files.append(os.path.join(current_root, file_name))

        return sorted(files)

    def _frame_from_xml_name(self, xml_file):

        name = os.path.splitext(os.path.basename(xml_file))[0]
        match = re.search(r'(\d+)$', name)

        if match is None:
            return None

        return int(match.group(1))

    def _clean_name(self, name):

        name = str(name).lower()
        name = re.sub(r'-\s*m\s*-\s*\d+', '', name)
        name = re.sub(r'[^a-z0-9]+', '', name)

        return name

    def _column_index(self, cell_ref):

        letters = re.sub(r'[^A-Z]', '', cell_ref.upper())
        index = 0

        for letter in letters:
            index = index * 26 + ord(letter) - ord('A') + 1

        return index - 1

    def _to_number(self, value):

        if value is None:
            return None

        try:
            number = float(value)
        except (TypeError, ValueError):
            return value

        if number.is_integer():
            return int(number)

        return number

    def _to_int(self, value):

        if value is None or value == '':
            return None

        try:
            return int(float(value))
        except (TypeError, ValueError):
            return None

    def _to_float(self, value):

        if value is None or value == '':
            return None

        try:
            return float(value)
        except (TypeError, ValueError):
            return None


if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default='Dataset')
    parser.add_argument('--output', type=str, default=None)
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--train-ratio', type=float, default=0.7)
    parser.add_argument('--val-ratio', type=float, default=0.1)
    parser.add_argument('--keep-bad-remarks', action='store_true')
    parser.add_argument('--keep-duplicates', action='store_true')
    parser.add_argument('--no-clamp', action='store_true')
    parser.add_argument('--split-method', type=str, default='paper40', choices=['paper', 'paper40', 'group'])
    parser.add_argument('--audit-json', type=str, default=None)
    args = parser.parse_args()

    preprocessor = BilliardsPreprocessor(root=args.root, output_path=args.output)
    data = preprocessor.preprocess(train_ratio=args.train_ratio,
                                   val_ratio=args.val_ratio,
                                   seed=args.seed,
                                   drop_bad_remarks=not args.keep_bad_remarks,
                                   deduplicate=not args.keep_duplicates,
                                   clamp_coordinates=not args.no_clamp,
                                   split_method=args.split_method,
                                   audit_json=args.audit_json,
                                   save=True)

    print('saved:', preprocessor.output_path)
    print('samples:', data['x'].shape[0])
    print('x:', tuple(data['x'].shape))
    print('x_paper:', tuple(data['x_paper'].shape))
    print('split_method:', data['split_method'])
    print('train:', len(data['split_indices']['train']))
    print('val:', len(data['split_indices']['val']))
    print('test:', len(data['split_indices']['test']))
    if args.audit_json is not None:
        print('audit_json:', args.audit_json)
