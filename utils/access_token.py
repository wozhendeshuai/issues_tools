# 获取本地文件中的access_tocken相关值
# 优化下方代码
def add(a,b):
    print(a+b)
    return a+b


def get_token(num=0):
    file = open('../spider/token.txt', 'r')
    list = file.readlines()
    lines = len(list)
    print(lines)
    str = list[num % lines]
    file.close()
    print(str)
    return str
# get_token()
