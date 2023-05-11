import pymysql as db
import requests
from utils.time_utils import time_reverse
from utils.access_token import get_token
from utils.url_utils import findUrlJsonCount
import time
import json

# 给下方代码加上注释

# 封装成一个方法，让他方便外部调用
def get_repo_info(index, num, owner_name, repo_name, headers):
    # repo url拼接
    repo_url = "https://api.github.com/repos/" + owner_name + "/" + repo_name
    # 组织url 拼接，主要是为了找多少人 page后面的用拼接来确定到底有多少人，省的去页面爬了，太麻烦。
    org_url = "https://api.github.com/orgs/" + owner_name + "/members"
    print(repo_url)
    print(org_url)

    try:
        repo_r = requests.get(repo_url, headers=headers)
        print("repo_url: " + repo_url + "  Status Code:", repo_r.status_code)
    except Exception as e:
        # 如果发生错误则回滚
        print("网络连接失败: repo_url: ", repo_url, "org_url: ", org_url)
        filename = 'repo_exception.csv'
        write_file(index, repo_name + "-" + owner_name,
                   str(e) + ("网络连接失败: user_name: " + repo_name + "owner_name: " + owner_name),
                   filename)
        print(e)
        num = num + 1
        access_token = get_token(num)
        headers = {
            'Authorization': 'token ' + access_token
        }
        time.sleep(10)

        # 如果返回的状态码以2开头，则说明正常此时去写入到数据库中即可
    if repo_r.status_code >= 200 and repo_r.status_code < 300:
        repo_json_str = repo_r.json()
        length_repo_json = len(repo_json_str)
        print("repo_json_str:", length_repo_json)
        # 基础数据
        repo_id = repo_json_str["id"]
        full_name = repo_json_str["full_name"]
        owner_type = repo_json_str["owner"]["type"]
        created_at = repo_json_str["created_at"]
        updated_at = repo_json_str["updated_at"]
        pushed_at = repo_json_str["pushed_at"]
        watchers = repo_json_str["subscribers_count"]
        stars = repo_json_str["stargazers_count"]
        # 获取主要language
        use_language = repo_json_str["language"]
        forks_count = repo_json_str["forks"]
        # 获取所有languages所对应的json
        languages_r = requests.get(repo_json_str["languages_url"], headers=headers)
        languages_json = json.dumps(languages_r.json())
        # 获取project_domain并转换为json格式
        topics = {}
        for i in range(0, repo_json_str["topics"].__len__()):
            topics[i] = repo_json_str["topics"][i]

        project_domain = json.dumps(topics)
        # 获取contributor的数量
        contributor_num = findUrlJsonCount(repo_json_str["contributors_url"], headers)
        team_size = None
        # team_size的数量统计
        if repo_json_str["owner"]["type"].__eq__("Organization"):
            team_size = findUrlJsonCount(org_url, headers)

    else:
        filename = 'repo_exception.csv'
        write_file(index, "user",
                   ("第" + str(index) + "行连接有问题: " + "repo_name:" + repo_name),
                   filename)


if __name__ == '__main__':
    # 此部分可修改，用于控制进程

    owner_name = "apache"  # "django"#"apache"#"apache"#"apache"#"apache"#"helix-editor"#"hibernate"#"Homebrew"#"apache"#"Ipython" #"apache"  # "Katello"#"kubernetes"#"mdn"#"openzipkin"#"laravel" #"apache"#  # "spring-projects"  # "symfony"#"rails"#"angular" #"tensorflow"
    repo_name = "guacamole-client"  # "django"#"dubbo"#"flume"#"groovy"#"guacamole-client" #"helix"#"hibernate-orm"#"homebrew-cask"#"incubator-heron"#"Ipython"#"kafka"  # "Katello"#"kubernetes"#"kuma"#"zipkin"#"laravel" #"lucene-solr"#  # "spring-framework"  # "spring-boot" #"symfony"#"rails"#"angular.js"#"tensorflow"

    access_token = get_token()
    # 拼接多个请求头
    headers = {
        'Authorization': 'Bearer ' + access_token,
        'X-GitHub-Api-Version': '2022-11-28'
    }

    get_repo_info(index, max_pr_num, owner_name, repo_name, headers)
