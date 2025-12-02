import logging
from ftplib import FTP
import os

FTP_IP = ""
FTP_USER = ""
FTP_PASS = ""
FTP_DIR = "/cams/All-Sky/lapup/2023/01"

logger = logging.getLogger(__name__)


def list_recursive(ftp, remotedir, file_paths):
    ftp.cwd(remotedir)
    # logger.debug(f"Remote CWD: {remotedir}")
    for entry in ftp.mlsd():
        remotepath = remotedir + "/" + entry[0]
        if entry[1]["type"] == "dir":
            list_recursive(ftp, remotepath, file_paths)
        elif entry[1]["type"] == "file":
            file_paths.append(remotepath)


def get_remote_paths(ftp_dir=FTP_DIR):
    logger.debug(f"Getting remote paths for {FTP_DIR}/*")
    all_paths = []
    with FTP(FTP_IP, FTP_USER, FTP_PASS) as ftp:
        list_recursive(ftp, ftp_dir, all_paths)
    logger.info(f"Found {len(all_paths)} files in {FTP_DIR}/*")
    return all_paths


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


def download_multiple(remote_files):
    logger.info(f"Downloading {len(remote_files)} from FTP...")
    with FTP(FTP_IP, FTP_USER, FTP_PASS) as ftp:
        for remote_path in remote_files:
            local_path = remote_path[1:]  # remove first / from path
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            with open(local_path, "wb") as f:
                ftp.retrbinary("RETR " + remote_path, f.write)
            logger.debug(f"Downloaded file: {local_path}")
    logger.info(f"Downloaded {len(remote_files)} files")


def upload(local_path, remote_path):
    with FTP(FTP_IP, FTP_USER, FTP_PASS) as ftp:
        with open(local_path, "rb") as f:
            ftp.storbinary(f"STOR {remote_path}", f)
    logger.info(f"Uploaded {local_path} to {remote_path}")
