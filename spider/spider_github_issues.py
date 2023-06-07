import pymysql as db
import requests
from utils.time_utils import time_reverse
from utils.access_token import get_token
from utils.url_utils import findUrlJsonCount
from utils.file_utils import write_exception_file
from utils.file_utils import write_excel
import time
import json

import re

ILLEGAL_CHARACTERS_RE = re.compile(r'[\000-\010]|[\013-\014]|[\016-\037]')

# 给下方代码加上注释


# 封装成一个方法，让他方便外部调用
def get_issues_info(page_num, issues_id, max_issues_id, owner_name, repo_name, headers):
    # repo url拼接
    list_issues_url = "https://api.github.com/repos/" + owner_name + "/" + repo_name + "/issues?state=all&direction=asc&per_page=100&page=" + str(
        page_num)

    print(list_issues_url)
    max_page = max_issues_id / 100 + 1
    while page_num <= max_page:
        try:
            print("=========================开始获取第"+ str(page_num)+"页issues===========================================")
            list_issues_request = requests.get(list_issues_url, headers=headers)
            print("list_issues_url: " + list_issues_url + "  Status Code:", list_issues_request.status_code)
        except Exception as e:
            # 如果发生错误则回滚
            print("网络连接失败: list_issues_url: ", list_issues_url)
            filename = 'repo_exception.csv'
            write_exception_file(issues_id, repo_name + "-" + owner_name,
                                 str(e) + ("网络连接失败: user_name: " + repo_name + "owner_name: " + owner_name), filename)
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
            for issues_index in range(0, length_list_issues_json):
                print( "=========================开始获取第" + str(page_num) + "页,第"+str(issues_index)+"个issues==============================================================")
                # 从json中提取数据
                issues_number = list_issues_json[issues_index]['number']==None and " " or list_issues_json[issues_index]['number']
                issues_title = list_issues_json[issues_index]['title']==None and " " or list_issues_json[issues_index]['title']
                issues_user_name = list_issues_json[issues_index]['user']['login']==None and " " or list_issues_json[issues_index]['user']['login']
                issues_user_id = list_issues_json[issues_index]['user']['id']==None and " " or list_issues_json[issues_index]['user']['id']
                issues_body = list_issues_json[issues_index]['body']==None and " " or list_issues_json[issues_index]['body']
                issues_created_at = list_issues_json[issues_index]['created_at']==None and " " or list_issues_json[issues_index]['created_at']
                issues_text = 'number: ' + str(issues_number) + '\ntitle:' + issues_title + \
                              '\ncreated_at:' + issues_created_at + \
                              '\nuser:' + str(issues_user_id) + '-' + issues_user_name + '\nbody:' + issues_body
                issues_labels_list = list_issues_json[issues_index]['labels']
                issues_labels = []
                if len(issues_labels_list) > 0:
                    issues_labels.append(issues_labels_list[0]['name'])
                write_excel(repo_name, owner_name + "-" + repo_name + "-data.xlsx", issues_text, issues_labels)
                issues_text = ILLEGAL_CHARACTERS_RE.sub(r'', issues_text)
                print(issues_text, issues_labels)
            # 当顺利解析后切换到下一页
            page_num = page_num + 1
            list_issues_url = "https://api.github.com/repos/" + owner_name + "/" + repo_name + "/issues?state=all&direction=asc&per_page=100&page=" + str(
                page_num)

        else:
            filename = 'list_issues_exception.csv'
            write_exception_file(index, "user", ("第" + str(index) + "行连接有问题: " + "repo_name:" + repo_name), filename)


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
