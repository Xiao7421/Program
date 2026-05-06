import hashlib
import os.path
from logging import error
from xml.dom.minidom import Document

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document
from utils.logger_handler import logger


def get_file_md5_hex(file_path):
    if not os.path.exists(file_path):
        logger.error(f"[md5计算]文件{file_path}不存在")
        return
    if not  os.path.isfile(file_path):
        logger.error(f"[md5计算]路径{file_path}不是文件")
        return

    md5_obj = hashlib.md5()
    chunksize= 4096
    try:
        with open(file_path,'rb') as f:
            while chunk :=f.read(chunksize):
                md5_obj.update(chunk)
            md5_hex = md5_obj.hexdigest()
            return  md5_hex
    except Exception as e:

        logger.error(f"计算文件{file_path}md5失败.str{e}")
        return  None
def listdir_with_allowed_type(path:str,allowed_types:tuple[str]):    #返回文件夹的文件列表
    files =[]
    if not os.path.isdir(path):
        logger,error(f"[listdir_with_allowed_types]{path}不是文件夹")
        return allowed_types
    for f in os.listdir(path):
        if f.endswith(allowed_types):
            files.append(os.path.join(path,f))
    return  tuple(files)

def pdf_loader(filepath:str,password=None) ->list[Document]:
    return PyPDFLoader(filepath,password=password).load()


def txt_loader(filepath:str,password=None) ->list[Document]:
    return TextLoader(filepath,encoding="utf-8").load()
