import threading

import pymysql as db
import requests
from utils.time_utils import time_reverse
from utils.access_token import get_token
from utils.url_utils import findUrlJsonCount
import time
import json
from spider.sql_thread import execute_query, execute_select_query


# 控制爬取流程
def process(owner_name, repo_name, max_issues_number):
    table_name = owner_name + "_" + repo_name + "_issues"

    result = execute_select_query(f"SELECT * FROM {table_name}")
    if result[0][0] is None:
        db_max_number = 0
    else:
        print("=====" + result[0].__str__())


if __name__ == '__main__':
    owner_name_list = ["apache", "apache", "apache", "tensorflow", "pengzhile", "TransformerOptimus", "openzipkin"]
    repo_name_list = ["dubbo", "superset", "echarts", "tensorflow", "pandora", "SuperAGI", "zipkin"]
    max_issues_number_list = [12698, 24631, 18866, 61220, 767, 183, 3547]
    threads = []
    process("apache", "dubbo", 12698)
    # for i in range(max_issues_number_list.__len__()):
    #     t = threading.Thread(target=process, args=(owner_name_list[i], repo_name_list[i], max_issues_number_list[i]))
    #     threads.append(t)
    #     t.start()
    #
    # for t in threads:
    #     t.join()
