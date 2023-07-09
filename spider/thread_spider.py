import mysql.connector
import threading

import pymysql as db
import requests
from utils.time_utils import time_reverse
from utils.access_token import get_token
from utils.url_utils import findUrlJsonCount
import time
import json

import re

# 创建数据库
mydb = mysql.connector.connect(
    host="localhost",
    user="root",
    password="root",
    database="github_issues_db"
)

mycursor = mydb.cursor()

access_token = get_token()
# 拼接多个请求头
headers = {
    'Authorization': 'Bearer ' + access_token,
    'X-GitHub-Api-Version': '2022-11-28'
}




def create_table(table_name):
    mycursor.execute("SHOW TABLES")

    tables = mycursor.fetchall()

    table_exists = False

    for table in tables:
        if table[0] == table_name:
            table_exists = True
            break

    if table_exists:
        print("名为：" + table_name + "的表已存在")
    else:
        mycursor.execute(
            f"CREATE TABLE {table_name} (issues_number INT AUTO_INCREMENT PRIMARY KEY, issues_text LONGTEXT, labels VARCHAR(255))")
        print("Table created successfully")


# 向表中插入issues_text和labels
def write_to_table(repo_name, issues_number, issues_text, labels):
    table_name = repo_name + "_issues"

    sql = f"INSERT INTO {table_name} (issues_number, issues_text, labels) VALUES (%s, %s, %s)"
    val = (issues_number, issues_text, str(labels))
    mycursor.execute(sql, val)
    mydb.commit()
    print("在表中", table_name + "插入", issues_number, "的issues_text和labels")


# 封装成一个方法，让他方便外部调用
def batch_set_issues_info(db_max_number, max_issues_number, owner_name, repo_name):
    # 获取当前页面
    page_num = int(db_max_number / 100) + 1
    # repo url拼接
    list_issues_url = "https://api.github.com/repos/" + owner_name + "/" + repo_name + "/issues?state=all&direction=asc&per_page=100&page=" + str(
        page_num)

    print(list_issues_url)
    max_page = int(max_issues_number / 100 + 1)
    while page_num <= max_page:
        try:
            print("=======================开始获取第" + str(page_num) + "页issues================================")
            list_issues_request = requests.get(list_issues_url, headers=headers)
            print("list_issues_url: " + list_issues_url + "  Status Code:", list_issues_request.status_code)
        except Exception as e:
            # 如果发生错误则回滚
            print("网络连接失败: list_issues_url: ", list_issues_url)

            # 如果返回的状态码以2开头，则说明正常此时去写入到数据库中即可
        if list_issues_request.status_code >= 200 and list_issues_request.status_code < 300:
            list_issues_json = list_issues_request.json()
            length_list_issues_json = len(list_issues_json)
            print("length_list_issues_json:", length_list_issues_json)
            for issues_index in range(0, length_list_issues_json):
                print("================开始获取第" + str(page_num) + "页,第" + str(issues_index) + "个issues===========")
                # 从json中提取数据
                issues_number = list_issues_json[issues_index]['number'] == None and " " or \
                                list_issues_json[issues_index]['number']
                if issues_number <= db_max_number:
                    print("这个issues已经爬取过了，跳过，issues_number:", issues_number)
                    continue
                issues_title = list_issues_json[issues_index]['title'] == None and " " or \
                               list_issues_json[issues_index]['title']
                issues_user_name = list_issues_json[issues_index]['user']['login'] == None and " " or \
                                   list_issues_json[issues_index]['user']['login']
                issues_user_id = list_issues_json[issues_index]['user']['id'] == None and " " or \
                                 list_issues_json[issues_index]['user']['id']
                issues_body = list_issues_json[issues_index]['body'] == None and " " or list_issues_json[issues_index][
                    'body']
                issues_created_at = list_issues_json[issues_index]['created_at'] == None and " " or \
                                    list_issues_json[issues_index]['created_at']
                issues_text = 'number: ' + str(issues_number) + '\ntitle:' + issues_title + \
                              '\ncreated_at:' + issues_created_at + \
                              '\nuser:' + str(issues_user_id) + '-' + issues_user_name + '\nbody:' + issues_body
                issues_labels_list = list_issues_json[issues_index]['labels']
                issues_labels = []
                if len(issues_labels_list) > 0:
                    for issues_labels_index in range(0, len(issues_labels_list)):
                        issues_labels.append(issues_labels_list[issues_labels_index]['name'])
                print(issues_text, issues_labels)
                write_to_table(repo_name, issues_number, issues_text, issues_labels)
            # 当顺利解析后切换到下一页
            page_num = page_num + 1
            list_issues_url = "https://api.github.com/repos/" + owner_name + "/" + repo_name + "/issues?state=all&direction=asc&per_page=100&page=" + str(
                page_num)

        else:
            # 如果返回的状态码不是以2开头，则说明发生了错误，此时需要打印错误信息
            print("list_issues_url: " + list_issues_url + "  Status Code:", list_issues_request.status_code)
            print("list_issues_url: " + list_issues_url + "  Error Response:", list_issues_request.text)
            # 如果发生错误则回滚
            print("网络连接失败: list_issues_url: ", list_issues_url)
            break


# 控制爬取流程
def process(owner_name, repo_name, max_issues_number):
    table_name = repo_name + "_issues"

    create_table(table_name)

    mycursor.execute(f"SELECT MAX(issues_number) FROM {table_name}")
    result = mycursor.fetchone()
    db_max_number = result[0] if result[0] else 0

    if db_max_number < max_issues_number:
        batch_set_issues_info(db_max_number, max_issues_number, owner_name, repo_name)

    mycursor.close()
    mydb.close()


if __name__ == '__main__':

    # owner_name_list = ["pengzhile","TransformerOptimus"]
    # repo_name_list= ["pandora","SuperAGI"]
    # max_issues_number_list = [767,183]
    threads = []

    # for i in range(max_issues_number_list.__len__()):
    #     t = threading.Thread(target=process, args=(owner_name_list[i], repo_name_list[i], max_issues_number_list[i],mycursor))
    #     threads.append(t)
    #     t.start()
    #
    # for t in threads:
    #     t.join()
    owner_name = "apache"#"tensorflow"
    repo_name = "dubbo"#"tensorflow"
    max_issues_number = 12465
    process(owner_name, repo_name, max_issues_number)
# concurrent_write_to_table("input_parameter", "This is a sample issue", "bug", 4)
