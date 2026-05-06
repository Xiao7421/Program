import random
from utils.config_handler import agent_conf
from langchain_core.tools import tool
from rag.rag_service import  RagSummarizeService
from utils.path_tool import get_abs_path
from utils.logger_handler import logger
import  os

rag = RagSummarizeService()
@tool(description="从向量存储中检索参考资料")
def rag_summarize(query:str)->str:
    return rag.rag_summarize(query)

@tool(description="获取天气信息")
def get_weather(query:str)->str:
    return f"{query}天气晴朗,天气很好"
external_data = {}
@tool(description="获取城市信息")
def get_user_location()->str:
        return random.choice(["北京", "上海", "广州", "深圳", "成都", "杭州", "南京", "武汉", "西安", "重庆"])

@tool(description="获取用户id")
def get_user_id()->str:
    return  random.choice(["1001", "1002", "1003", "1004", "1005", "1006", "1007", "1008", "1009", "1010"])
@tool(description="获取月份")
def get_current_month()->str:
    return random.choice(["2025-01", "2025-02", "2025-03", "2025-04", "2025-05", "2025-06", "2025-07", "2025-08", "2025-09", "2025-10", "2025-11", "2025-12"])


def generate_external_data()->str:
    if not external_data:
      external_data_path =  get_abs_path(agent_conf["external_data_path"] )
      if not os.path.exists(external_data_path):
          raise FileNotFoundError("外部文件不存在")

      with open(external_data_path, "r", encoding="utf-8") as f:
          for line in f.readlines():
              arr:list[str] = line.strip().split(",")
              user_id = arr[0].replace('"',"")
              feature  = arr[1].replace('"',"")
              efficiency = arr[2].replace('"', "")
              consumables = arr[3].replace('"',"")
              comparison = arr[4].replace('"', "")
              time = arr[5].replace('"', "")
              if user_id not in external_data:
                external_data[user_id] = {}
                external_data[user_id][time] = {
                    "特征": feature,
                    "效率": efficiency,
                    "耗材": consumables,
                    "对比": comparison,
                }


@tool(description="获取特定用户的使用记录")
def fetch_external_data(user_id,month)->str:
    generate_external_data()

    try:
        return external_data[user_id][month]
    except KeyError:
        logger.warning(f"[fetch_external_data]未能检索到用户:{user_id}的{month}月数据")
        return '无数据'

if __name__ == '__main__':
    print(fetch_external_data("1001", "2025-01"))
