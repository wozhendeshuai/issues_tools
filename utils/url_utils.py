import requests
# 统计url中含有的元素数量
def findUrlJsonCount(url_str, headers):
    url_str = url_str + "?per_page=100&anon=true&page="
    print(url_str)
    page = 1
    count = 0
    while 1:
        temp_url_str = url_str + page.__str__()
        print(temp_url_str)
        url_r = requests.get(temp_url_str, headers=headers)
        url_json = url_r.json()
        if len(url_json) < 100:
            count = count + len(url_json)
            return count
        else:
            count = count + 100
            page = page + 1
    return count