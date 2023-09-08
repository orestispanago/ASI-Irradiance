import logging
from ftplib import FTP

FTP_IP = ""
FTP_USER = ""
FTP_PASS = ""
FTP_DIR = "/cams/All-Sky/lapup"

logger = logging.getLogger(__name__)


def get_last_file_stats(ftp, remote_dir):
    """Enters last FTP directory recursively and gets last file stats"""
    ftp.cwd(remote_dir)
    folders, files = [], []
    for entry in ftp.mlsd():
        if entry[1]["type"] == "dir":
            folders.append(entry)
        elif entry[1]["type"] == "file":
            files.append(entry)
    folders = sorted(folders)
    if len(folders) > 0:
        last_folder = folders[-1]
        remote_dir = f"{remote_dir}/{last_folder[0]}"
        return get_last_file_stats(ftp, remote_dir)
    files = sorted(files)
    last_file = files[-1]
    fname = last_file[0]
    modified = last_file[1]["modify"]
    return remote_dir, fname, modified


def get_last_file_path():
    with FTP(FTP_IP, FTP_USER, FTP_PASS) as ftp:
        dirname, fname, modified = get_last_file_stats(ftp, FTP_DIR)
    return f"{dirname}/{fname}"


def download(remote_path, local_path):
    with FTP(FTP_IP, FTP_USER, FTP_PASS) as ftp:
        with open(local_path, "wb") as f:
            ftp.retrbinary("RETR " + remote_path, f.write)
        logger.info(f"Downloaded {remote_path} to {local_path}")


def upload(local_path, remote_path):
    with FTP(FTP_IP, FTP_USER, FTP_PASS) as ftp:
        with open(local_path, "rb") as f:
            ftp.storbinary(f"STOR {remote_path}", f)
    logger.info(f"Uploaded {local_path} to {remote_path}")
