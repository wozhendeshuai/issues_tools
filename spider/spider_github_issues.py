import pymysql as db
import requests
from utils.time_utils import time_reverse
from utils.access_token import get_token
from utils.url_utils import findUrlJsonCount
from utils.exception_handdle import write_file
import time
import json


# 给下方代码加上注释

# 封装成一个方法，让他方便外部调用
def get_issues_info(page_num, issues_id, max_issues_id, owner_name, repo_name, headers):
    # repo url拼接
    list_issues_url = "https://api.github.com/repos/" + owner_name + "/" + repo_name + "/issues?state=all&direction=asc&per_page=100&page="

    print(list_issues_url)
    max_page = max_issues_id / 100 + 1
    while page_num <= max_page:
        try:
            list_issues_request = requests.get(list_issues_url, headers=headers)
            print("list_issues_url: " + list_issues_url + "  Status Code:", list_issues_request.status_code)
        except Exception as e:
            # 如果发生错误则回滚
            print("网络连接失败: list_issues_url: ", list_issues_url)
            filename = 'repo_exception.csv'
            write_file(issues_id, repo_name + "-" + owner_name,
                       str(e) + ("网络连接失败: user_name: " + repo_name + "owner_name: " + owner_name),
                       filename)
            print(e)
            num = num + 1
            access_token = get_token(num)
            # 拼接多个请求头
            headers = {
                'Authorization': 'Bearer ' + access_token,
                'X-GitHub-Api-Version': '2022-11-28'
            }
            time.sleep(10)

            # 如果返回的状态码以2开头，则说明正常此时去写入到数据库中即可
        if list_issues_request.status_code >= 200 and list_issues_request.status_code < 300:
            list_issues_json = list_issues_request.json()
            length_list_issues_json = len(list_issues_json)
            print("length_list_issues_json:", length_list_issues_json)

            # 当顺利解析后切换到下一页
            page_num = page_num + 1
        else:
            filename = 'list_issues_exception.csv'
            write_file(index, "user",
                       ("第" + str(index) + "行连接有问题: " + "repo_name:" + repo_name),
                       filename)


if __name__ == '__main__':
    # 此部分可修改，用于控制进程

    owner_name = "tensorflow"
    repo_name = "tensorflow"

    access_token = get_token()
    # 拼接多个请求头
    headers = {
        'Authorization': 'Bearer ' + access_token,
        'X-GitHub-Api-Version': '2022-11-28'
    }
    page_num = 1
    issues_id = 0
    max_issues_id = 60563
    get_issues_info(page_num, issues_id, max_issues_id, owner_name, repo_name, headers)
