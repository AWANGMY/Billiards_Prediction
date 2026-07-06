import argparse
import concurrent.futures
import html
import os
import re
import urllib.parse
import urllib.request
from collections import OrderedDict


OFFICIAL_FOLDERS = {
    'code': {
        'id': '1vqAJmJu7lbANcP6Y1poPiccdLhObMtja',
        'path': os.path.join('Baseline', 'code'),
    },
    'data_layouts': {
        'id': '1lgjKrR-5aNhOSth8xf7IigXRMkwt7wcS',
        'path': os.path.join('Dataset', 'data_layouts'),
    },
    'data_trajectories': {
        'id': '1COSGpcPbrZcgD2NHXOAn9-zbJAGn7alg',
        'path': os.path.join('Dataset', 'data _trajectories'),
    },
}


def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('--target',
                        choices=['code', 'data_layouts', 'data_trajectories', 'all'],
                        default='code')
    parser.add_argument('--output-root', default='.')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--extensions',
                        nargs='*',
                        default=None,
                        help='Optional file suffix allowlist, for example: --extensions .xml .xlsx')
    parser.add_argument('--workers',
                        type=int,
                        default=1,
                        help='Number of parallel file downloads. Folder listing remains sequential.')
    return parser.parse_args()


def urlopen_bytes(url):

    request = urllib.request.Request(
        url,
        headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read()


def list_drive_folder(folder_id):

    url = 'https://drive.google.com/drive/folders/' + folder_id
    page = urlopen_bytes(url).decode('utf-8', errors='ignore')

    items = OrderedDict()
    for match in re.finditer(r'data-id="([\w-]{20,})".{0,5000}?aria-label="([^"]+)".{0,5000}?'
                             r'<strong class="DNoYtb">([^<]+)</strong>',
                             page,
                             re.S):
        item_id = match.group(1)
        label = html.unescape(match.group(2))
        name = sanitize_name(html.unescape(match.group(3)))
        is_folder = 'folder' in label.lower()

        if item_id == folder_id or item_id.endswith('-0'):
            continue
        if item_id not in items:
            items[item_id] = {'id': item_id,
                              'name': name,
                              'is_folder': is_folder,
                              'label': label}

    return list(items.values())


def sanitize_name(name):

    name = name.strip()
    name = name.replace('/', '_')
    return name


def download_file(file_id, destination, overwrite=False, dry_run=False):

    if os.path.exists(destination) and not overwrite:
        print('skip existing:', destination)
        return

    print('download:', destination)
    if dry_run:
        return

    os.makedirs(os.path.dirname(destination), exist_ok=True)
    page = urlopen_bytes('https://drive.google.com/uc?export=download&id=' + file_id)

    if b'id="download-form"' in page:
        text = page.decode('utf-8', errors='ignore')
        action_match = re.search(r'<form[^>]+id="download-form"[^>]+action="([^"]+)"', text)
        inputs = re.findall(r'<input type="hidden" name="([^"]+)" value="([^"]*)"', text)
        if action_match is None:
            raise RuntimeError('Could not parse Google Drive confirmation page for ' + file_id)

        query = urllib.parse.urlencode([(key, html.unescape(value)) for key, value in inputs])
        page = urlopen_bytes(html.unescape(action_match.group(1)) + '?' + query)

    with open(destination, 'wb') as output_file:
        output_file.write(page)


def download_folder(folder_id,
                    destination,
                    overwrite=False,
                    dry_run=False,
                    extensions=None,
                    executor=None,
                    futures=None,
                    visited=None):

    if visited is None:
        visited = set()
    if folder_id in visited:
        return
    visited.add(folder_id)

    print('folder:', destination)
    if not dry_run:
        os.makedirs(destination, exist_ok=True)

    items = list_drive_folder(folder_id)
    if len(items) == 0:
        print('empty folder:', destination)
        return

    for item in items:
        item_path = os.path.join(destination, item['name'])
        if item['is_folder']:
            download_folder(item['id'], item_path,
                            overwrite=overwrite,
                            dry_run=dry_run,
                            extensions=extensions,
                            executor=executor,
                            futures=futures,
                            visited=visited)
        else:
            if extensions is not None and not has_allowed_extension(item['name'], extensions):
                print('skip extension:', item_path)
                continue
            if executor is None:
                download_file(item['id'], item_path,
                              overwrite=overwrite,
                              dry_run=dry_run)
            else:
                futures.append(executor.submit(download_file,
                                               item['id'],
                                               item_path,
                                               overwrite,
                                               dry_run))


def has_allowed_extension(name, extensions):

    lower_name = name.lower()
    return any(lower_name.endswith(extension.lower()) for extension in extensions)


def selected_targets(target):

    if target == 'all':
        return ['code', 'data_layouts', 'data_trajectories']
    return [target]


def main():

    args = parse_args()
    futures = []
    if args.workers > 1:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
            for target in selected_targets(args.target):
                folder = OFFICIAL_FOLDERS[target]
                destination = os.path.join(args.output_root, folder['path'])
                download_folder(folder['id'],
                                destination,
                                overwrite=args.overwrite,
                                dry_run=args.dry_run,
                                extensions=args.extensions,
                                executor=executor,
                                futures=futures)
            for future in concurrent.futures.as_completed(futures):
                future.result()
    else:
        for target in selected_targets(args.target):
            folder = OFFICIAL_FOLDERS[target]
            destination = os.path.join(args.output_root, folder['path'])
            download_folder(folder['id'],
                            destination,
                            overwrite=args.overwrite,
                            dry_run=args.dry_run,
                            extensions=args.extensions)


if __name__ == '__main__':
    main()
