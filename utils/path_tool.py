import  os
def get_project_root() -> str:
     current_file = os.path.abspath(__file__)
     current_dir = os.path.dirname(current_file) #获取文件的文件夹位置
     project_root = os.path.dirname(current_dir) #获取文件夹的上一级目录
     return project_root

def get_abs_path(relative:str):

    """
    根据相对路径获取绝对路径

    参数:
        relative (str): 相对路径字符串

    返回:
        str: 绝对路径字符串
    """
    project_root = get_project_root()  # 获取项目根目录
    return os.path.join(project_root,relative)  # 拼接项目根目录和相对路径，返回绝对路径
if __name__ == '__main__':
    print(get_abs_path('data/'))

