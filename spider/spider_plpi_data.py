import threading

import requests
from utils.access_token import get_token
import time
import json
from sql_thread import execute_query, execute_select_query
from concurrent.futures import ThreadPoolExecutor
import re
# 添加进度条
import tqdm

access_token = get_token()
# 拼接多个请求头
headers = {
    'Authorization': 'Bearer ' + access_token,
    'X-GitHub-Api-Version': '2022-11-28'
}

# 在控制台输出日志，给出当前的线程名称，所用的方法名称，以及当前的时间等信息
# 打印当前时间 '%Y-%m-%d %H:%M:%S'


print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))


def log_str(method_name, owner_name, repo_name, table_name, max_issues_number, db_max_number, message):
    reStr = "======" + " 时间:" + time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(
        time.time())) + " 线程名:" + threading.current_thread().name

    if method_name is not None:
        reStr = reStr + "  方法名:  " + str(method_name)
    if owner_name is not None:
        reStr = reStr + "  owner_name:  " + str(owner_name)
    if repo_name is not None:
        reStr = reStr + "  项目名:" + str(repo_name)
    if table_name is not None:
        reStr = reStr + "  数据表名:  " + str(table_name)
    if max_issues_number is not None:
        reStr = reStr + " 真实最大的issues编号:" + str(max_issues_number)
    if db_max_number is not None:
        reStr = reStr + " 数据库最大的issue编号:" + str(db_max_number)
    if message is not None:
        reStr = reStr + " message:" + message

    return reStr


def create_table(table_name):
    tables = execute_select_query("SHOW TABLES")
    table_exists = False

    for table in tables:
        if table[0].lower() == table_name.lower():
            table_exists = True
            break

    if table_exists:
        print(log_str("create_table", None, None, table_name, None, None, "名为：" + table_name + "的表已存在"))
    else:
        execute_query(
            f"CREATE TABLE {table_name} (issues_number INT ,owner_repo_name VARCHAR(255), issues_text LONGTEXT, labels VARCHAR(255))")
        print(log_str("create_table", None, None, table_name, None, None, "名为：" + table_name + "的表创建成功"))


# 向表中插入issues_text和labels
def write_to_table(table_name, owner_name, repo_name, issues_number, issues_text, labels):
    sql = f"INSERT INTO {table_name} (issues_number,owner_repo_name, issues_text, labels) VALUES (%s, %s, %s, %s)"
    val = (issues_number, owner_name + "_" + repo_name, str(issues_text), str(labels))
    execute_query(sql, val)
    print(log_str("write_to_table", None, repo_name, table_name, None, None,
                  "在表中" + table_name + "插入" + str(issues_number) + "的issues_text和labels"))


# 封装成一个方法，让他方便外部调用
def batch_set_issues_info(db_max_number, max_issues_number, owner_name, repo_name, table_name):
    # 获取当前页面
    page_num = int(db_max_number / 100) + 1
    # repo url拼接
    list_issues_url = "https://api.github.com/repos/" + owner_name + "/" + repo_name + "/issues?state=closed&direction=asc&per_page=100&page=" + str(
        page_num)

    print(list_issues_url)
    max_page = int(max_issues_number / 100 + 1)
    while page_num <= max_page:
        try:
            print(log_str("batch_set_issues_info", owner_name, repo_name, table_name, max_issues_number, db_max_number,
                          "开始获取第" + str(page_num) + "页issues"))
            list_issues_request = requests.get(list_issues_url, headers=headers)
            print(log_str("batch_set_issues_info", owner_name, repo_name, table_name, max_issues_number, db_max_number,
                          "获取第" + str(page_num) + "页issues成功"))
        except Exception as e:
            # 如果发生错误则回滚
            print(log_str("batch_set_issues_info", owner_name, repo_name, table_name, max_issues_number, db_max_number,
                          "获取第" + str(page_num) + "页issues失败"))

            # 如果返回的状态码以2开头，则说明正常此时去写入到数据库中即可
        if list_issues_request.status_code >= 200 and list_issues_request.status_code < 300:
            list_issues_json = list_issues_request.json()
            length_list_issues_json = len(list_issues_json)

            print(log_str("batch_set_issues_info", owner_name, repo_name, table_name, max_issues_number, db_max_number,
                          "length_list_issues_json:" + str(length_list_issues_json)))
            for issues_index in range(0, length_list_issues_json):
                print(log_str("batch_set_issues_info", owner_name, repo_name, table_name, max_issues_number,
                              db_max_number,
                              "开始获取第" + str(page_num) + "页,第" + str(issues_index) + "个issues"))
                # 从json中提取数据
                issues_number = list_issues_json[issues_index]['number'] == None and " " or \
                                list_issues_json[issues_index]['number']
                if issues_number <= db_max_number:
                    print(log_str("batch_set_issues_info", owner_name, repo_name, table_name, max_issues_number,
                                  db_max_number, "这个issues已经爬取过了，跳过，issues_number:" + str(issues_number)))
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
                # issues_text = 'number: ' + str(issues_number) + '\ntitle:' + issues_title + \
                #               '\ncreated_at:' + issues_created_at + \
                #               '\nuser:' + str(issues_user_id) + '-' + issues_user_name + '\nbody:' + issues_body
                issues_labels_list = list_issues_json[issues_index]['labels']
                issues_labels = []
                if len(issues_labels_list) > 0:
                    for issues_labels_index in range(0, len(issues_labels_list)):
                        issues_labels.append(issues_labels_list[issues_labels_index]['name'])

                issues_text = {}
                issues_text["number"] = issues_number
                issues_text["title"] = issues_title
                issues_text["created_at"] = issues_created_at
                issues_text["user"] = str(issues_user_id) + '-' + issues_user_name
                issues_text["body"] = issues_body
                issues_text["label"] = issues_labels
                issues_text = json.dumps(issues_text, ensure_ascii=False)
                print(log_str("batch_set_issues_info", owner_name, repo_name, table_name, max_issues_number,
                              db_max_number,
                              "issues_text:" + issues_text + "issues_labels:" + str(issues_labels)))
                write_to_table(table_name, owner_name, repo_name, issues_number, issues_text, issues_labels)
            # 当顺利解析后切换到下一页
            page_num = page_num + 1
            list_issues_url = "https://api.github.com/repos/" + owner_name + "/" + repo_name + "/issues?state=all&direction=asc&per_page=100&page=" + str(
                page_num)

        else:
            # 如果返回的状态码不是以2开头，则说明发生了错误，此时需要打印错误信息
            print(log_str("batch_set_issues_info", owner_name, repo_name, table_name, max_issues_number, db_max_number,
                          "list_issues_url: " + list_issues_url + "  Status Code:" + str(
                              list_issues_request.status_code) + "  Error Response:" + str(list_issues_request.text)))
            break


# 获取当前开源项目最大关闭的issues number
def get_closed_issues_max_num(owner_name, repo_name):
    result = execute_select_query(
        f"SELECT closed_issues_number FROM closed_issues_number_tbl where repo_name='{repo_name}' and owner_name='{owner_name}'")
    if len(result) == 0:
        closed_issues_number = 0
    else:
        closed_issues_number = int(result[0][0])
    print(log_str("get_closed_issues_max_num", owner_name, repo_name, "closed_issues_number_tbl", None, None,
                  "当前数据库已保存的最大issuesNumber 为：" + str(closed_issues_number)))
    # repo url拼接
    list_issues_url = "https://api.github.com/repos/" + owner_name + "/" + repo_name + "/issues?state=closed&direction=desc"

    try:
        list_issues_request = requests.get(list_issues_url, headers=headers)
        print(log_str("get_closed_issues_max_num", owner_name, repo_name, "closed_issues_number_tbl", None, None,
                      " list_issues_url: " + list_issues_url + " Status Code:" + str(list_issues_request.status_code)))
    except Exception as e:
        # 如果发生错误则回滚
        print(log_str("get_closed_issues_max_num", owner_name, repo_name, "closed_issues_number_tbl", None, None,
                      "网络连接失败: list_issues_url: " + list_issues_url))

    # 如果返回的状态码以2开头，则说明正常此时去写入到数据库中即可,并取出其中第一个json中的issue_number
    if list_issues_request.status_code >= 200 and list_issues_request.status_code < 300:
        list_issues_json = list_issues_request.json()
        length_list_issues_json = len(list_issues_json)

        # 从json中提取数据
        issues_number = list_issues_json[0]['number'] == None and " " or list_issues_json[0]['number']
        if issues_number <= closed_issues_number:
            return closed_issues_number
        else:
            closed_issues_number = issues_number
            if len(result) == 0:
                sql = f"INSERT INTO closed_issues_number_tbl (repo_name,owner_name,closed_issues_number) VALUES ('{repo_name}','{owner_name}',{closed_issues_number})"
                print(log_str("get_closed_issues_max_num", owner_name, repo_name, "closed_issues_number_tbl",
                              closed_issues_number, None, "插入的closed_issues_number :" + str(closed_issues_number)))
            else:
                # 更新数据库中issues_closed_number
                sql = f"Update closed_issues_number_tbl set closed_issues_number={closed_issues_number} where repo_name='{repo_name}' and owner_name='{owner_name}'"
                print(log_str("get_closed_issues_max_num", owner_name, repo_name, "closed_issues_number_tbl",
                              closed_issues_number, result[0][0],
                              "更新的closed_issues_number :" + str(closed_issues_number)))
            execute_query(sql)

            return closed_issues_number
    else:
        return closed_issues_number


# 控制爬取流程
def process(owner_name, repo_name):
    max_issues_number = get_closed_issues_max_num(owner_name, repo_name)
    table_name = "All_issues"
    create_table(table_name)
    result = execute_select_query(f"SELECT MAX(issues_number) FROM {table_name} where owner_repo_name='{owner_name +'_' + repo_name}'")
    if result[0][0] is None:
        db_max_number = 0
    else:
        db_max_number = int(result[0][0])
    print(log_str("process", owner_name, repo_name, table_name, max_issues_number, db_max_number, None))
    if db_max_number < max_issues_number:
        batch_set_issues_info(db_max_number, max_issues_number, owner_name, repo_name, table_name)


# 读取txt文件，将每一行的json数据，解析后提取其中的project信息，保存在列表中
def txt_file_process(file_path):
    project_list = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for each_line in f:
            project_list.append(json.loads(each_line)['project'])
    return project_list


# 拆解得到的project名称，将第一个下划线前面的部分作为org_name,后面的部分作为repo_name，但是可能会在repo_name中存在下划线
def get_org_and_repo(project_list):
    owner_name_list = []
    repo_list = []
    for each_project in project_list:
        each_project = each_project.split('_')
        owner_name_list.append(each_project[0])
        repo_list.append('_'.join(each_project[1:]))
    return owner_name_list, repo_list


if __name__ == '__main__':
    file_path = "/Users/jiajunyu/PycharmProjects/plpi/PLPI_GitHub_dataset.txt"
    project_list = txt_file_process(file_path)
    print(project_list)
    print(len(project_list))
    print(len(set(project_list)))
    owner_name_list, repo_name_list = get_org_and_repo(set(project_list))
    print(len(owner_name_list))
    print(len(repo_name_list))
    # 用线程池处理，一次并发8个线程，后续进入线程池等待
    pool = ThreadPoolExecutor(max_workers=8)
    for i in range(len(owner_name_list)):
        pool.submit(process, owner_name_list[i], repo_name_list[i])
    pool.shutdown(wait=True)
    print("爬取结束")
