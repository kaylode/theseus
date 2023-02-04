import os
import os.path as osp
import urllib.request as urlreq

import gdown
import os
from pathlib import Path

from theseus.base.utilities.loggers.observer import LoggerObserver

LOGGER = LoggerObserver.getLogger("main")


def download_from_drive(id_or_url, output, md5=None, quiet=False, cache=True):
    if id_or_url.startswith("http") or id_or_url.startswith("https"):
        url = id_or_url
    else:
        url = "https://drive.google.com/uc?id={}".format(id_or_url)

    if not cache:
        return gdown.download(url, output, quiet=quiet)
    else:
        return gdown.cached_download(url, md5=md5, quiet=quiet)


def download_from_url(url, root=None, filename=None):
    """Download a file from a url and place it in root.
    Args:
        url (str): URL to download file from
        root (str): Directory to place downloaded file in
        filename (str, optional): Name to save the file under. If None, use the basename of the URL
    """

    if root is None:
        root = "./.cache"
    root = os.path.expanduser(root)
    if not filename:
        filename = os.path.basename(url)
    fpath = os.path.join(root, filename)

    if osp.isfile(fpath):
        LOGGER.text("Load cache from " + fpath, level=LoggerObserver.INFO)
        return fpath

    os.makedirs(root, exist_ok=True)

    try:
        LOGGER.text("Downloading " + url + " to " + fpath, level=LoggerObserver.DEBUG)
        urlreq.urlretrieve(url, fpath)
    except (urlreq.error.URLError, IOError) as e:
        if url[:5] == "https":
            url = url.replace("https:", "http:")
            LOGGER.text(
                "Failed download. Trying https -> http instead.Downloading "
                + url
                + " to "
                + fpath,
                level=LoggerObserver.DEBUG,
            )
            urlreq.urlretrieve(url, fpath)

    return fpath


def download_from_wandb(filename, run_path, save_dir, rename=None, generate_id_text_file=False):

    import wandb
    
    try:
        path = wandb.restore(filename, run_path=run_path, root=save_dir)
        LOGGER.text("Successfully download {} from wandb run path {}".format(filename, run_path), level=LoggerObserver.INFO)
        
        # Save run id to wandb_id.txt
        if generate_id_text_file:
            wandb_id = osp.basename(run_path)
            with open(osp.join(save_dir, "wandb_id.txt"), "w") as f:
                f.write(wandb_id)
        
        if rename:
            new_name = str(Path(path.name).resolve().parent / rename)
            os.rename(Path(path.name).resolve(), new_name)
            LOGGER.text("Saved to {}".format(new_name), level=LoggerObserver.INFO)
            return new_name
        
        
        LOGGER.text("Saved to {}".format((Path(save_dir) / path.name).resolve()), level=LoggerObserver.INFO)
        return path.name
    except:
        LOGGER.text("Failed to download from wandb.", level=LoggerObserver.ERROR)
        return None
