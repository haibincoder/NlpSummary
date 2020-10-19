from gevent import monkey
from gevent.pool import Pool

# 猴子补丁，替换底层thread/socket实现非阻塞
monkey.patch_all()
import gevent
import requests


def func(url):
    print('GET: %s' % url)
    resp = requests.get(url)
    data = resp.text
    print('%d bytes received from %s.' % (len(data), url))
    result_list.append(data)


result_list = []
if __name__ == "__main__":
    # gevent实现协程
    gevent.joinall([
        gevent.spawn(func, 'https://www.python.org/'),
        gevent.spawn(func, 'https://www.yahoo.com/'),
        gevent.spawn(func, 'https://github.com/'),
    ])
    print(f'result length:{len(result_list)}')

    # gevent限制协程数量
    inputs = ['https://www.python.org/', 'https://www.yahoo.com/', 'https://github.com/', 'https://www.baidu.com',
              'https://www.qq.com', 'https://www.meituan.com', 'https://www.iqiyi.com']
    pool = Pool(5)
    threads = [pool.spawn(func, i) for i in inputs]
    gevent.joinall(threads)
    print(f'finish1, data length:{len(result_list)}')
