import urllib
import os
import shutil
import tarfile
import zipfile
import gzip
import urllib.request

def path_to_string(path):
    # Convert `PathLike` object to their string representation. 
    if isinstance(path, os.PathLike):
        return os.fspath(path)
    return path

def extract_archive(file_path, path='.', archive_format='auto'):
    """
    Extracts an archive if it matches a support format.

    :param file_path: Path to the archive file.
    :param path: Where to extract the archive file.
    :param archive_format: Archive format to try for extracting the file.
    """
    if archive_format is None:
        return False, ''
    if archive_format == 'auto':
        archive_format = ['tar', 'zip', 'gz']
    if isinstance(archive_format, str):
        archive_format = [archive_format]

    file_path = path_to_string(file_path)
    path = path_to_string(path)

    open_fn = None
    for archive_type in archive_format:
        if archive_type == 'tar':
            open_fn = tarfile.open
        if archive_type == 'zip':
            open_fn = zipfile.ZipFile
        if archive_type == 'gz':
            open_fn = gzip.open
        else:
            return False, ''

    with open_fn(file_path) as archive:
        try:
            if zipfile.is_zipfile(file_path):
                # Zip archive.
                archive.extractall(path)
                return True, path
            elif tarfile.is_tarfile(file_path):
                # Tar archive.
                archive.extractall(path)
                return True, path
            else:
                # GZ archive.
                obj_file_path = os.path.join(path, os.path.basename(file_path).removesuffix('.gz'))
                with open(obj_file_path, 'wb') as file:
                    shutil.copyfileobj(archive, file)
                return True, obj_file_path
        except (tarfile.TarError, RuntimeError, KeyboardInterrupt):
            if os.path.exists(path):
                if os.path.isfile(path):
                    os.remove(path)
                else:
                    shutil.rmtree(path)
            raise

def get_file(origin_url=None,
             dest_file_name=None,
             cache_dir=None,
             extract=True,
             archive_format='auto',
             force_download=False):
    '''
    Download a file from a URL if it not already in the cache.
    
    By default, the file at the url `origin_url` is downloed to the cache dir `./temp/`, and
    given the filename `dest_file_name`. The final location of a file named `example.txt` would
    therefore be at `./temp/example.txt`.
    
    Example: path_to_download_file = get_file(origin_url='...', extract=True)
    
    Args:
    
    origin_url: Original URL of the file.
    dest_file_name: Name of the file. If absolute path, e.g. '/path/to/file.txt' is specified,
                    the file will be saved at that location. If `None`, the name of the file at
                    `origin` and default directory will be used.
    cache_dir: Location to store cached files, when None it defaults './temp/'.
    extract: True tries extracting the file as an archive, like tar or zip.
    archive_format: Archive format to try for extracting the file. Options are `auto`, `tar`,
                    `gz` and `zip`.
    force_download: If `True` the file will always be re-downloaded regardless of the cache state.
    
    Return:
    
        Path to the downloaded file.
    '''
    if origin_url is None:
        raise ValueError('Please specify the `origin_url` argument (URL of the file to download).')
    if cache_dir is None:
        cache_dir = os.path.join(os.curdir, 'temp')
    os.makedirs(cache_dir, exist_ok=True)

    file_name = path_to_string(dest_file_name)
    if not file_name:
        file_name = os.path.basename(origin_url)
        if not file_name:
            raise ValueError('Cannot parse the file anme for the origin provided.'
                             'Please specify the `dest_file_name` as the input param.')    
    file_path = os.path.join(cache_dir, file_name)
    if force_download:
        download = True
    elif os.path.exists(file_path):
        download = False
    else:
        download = True
    if download:
        try:
            urllib.request.urlretrieve(origin_url, file_path)
        except (Exception, KeyboardInterrupt):
            if os.path.exists(file_path):
                os.remove(file_path)
            raise
    
    dest_file_path = ''
    if extract:
        status, dest_file_path = extract_archive(file_path, cache_dir, archive_format)
        if not status:
            raise Exception('Could not extract archieve')
    # Return extracted file_path if we extracted an archive.
    return dest_file_path
