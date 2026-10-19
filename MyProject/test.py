


# %%===========================================================================
%reset -f

import random
import numpy as np

# 방법 1: random.randint()를 사용하여 100번 생성
# random_numbers = [random.randint(1, 6) for _ in range(100)]
# print(f"생성된 숫자 개수: {len(random_numbers)}")
# print(f"첫 10개: {random_numbers[:10]}")

# 방법 2: random.choices()를 사용 (더 효율적)
random_numbers2 = random.choices(range(1, 7), k=100)
print(f"\n생성된 숫자 개수: {len(random_numbers2)}")
print(f"첫 10개: {random_numbers2[:10]}")

print(f'\n===> 평균값 \n{np.mean(random_numbers2)}\n')


# 각 숫자의 빈도 확인
# from collections import Counter
# frequency = Counter(random_numbers)
# print(f"\n각 숫자의 빈도:")
# for num in sorted(frequency.keys()):
#     print(f"  {num}: {frequency[num]}번")





# %%===========================================================================
%reset -f

from functools import reduce

primes = [2, 3, 5, 7, 11, 13]

def product(*numbers):
    p = reduce(lambda x, y: x * y, numbers)
    return p

print(product(*primes))
# 30030

print(product(primes))
# [2, 3, 5, 7, 11, 13] 

print(f'aaa : {'bbb'}')


# %%===========================================================================
%reset -f

# def a(**args):
# def a(args):
def a(param_b: str, param_a : int):
    print (f'param_a: {param_a}, param_b:{param_b}')

args_list = {
    'param_a': 100,
    'param_c': 300,
    'param_b': 'xxx'
}


# a (**args)
a (**args_list)


# %%===========================================================================
%reset -f

import time

def longtime_job():
    print("job start")
    time.sleep(1)
    return "done"

# generator 표현식을 이용해서 generator를 만듦
list_job = (longtime_job() for i in range(5))

print(next(list_job)) 
print(next(list_job)) 
print(next(list_job)) 
print(next(list_job)) 

# %%===========================================================================
import time

def longtime_job():
    print("job start")
    time.sleep(1)
    return "done"

list_job = iter([longtime_job() for i in range(5)])

print(next(list_job)) 
# print(next(list_job)) 
# print(next(list_job)) 


# %%===========================================================================

def echo():
    while True:
        received = yield
        print(received)

# 제너레이터 객체 생성
gen = echo()

# 제너레이터 초기화
next(gen)

# 값을 보내고 출력
gen.send('Hello')  # 출력: Hello
gen.send('World')  # 출력: World


# %%===========================================================================

# This script is available in scripts/consult_environments.py
import sinergym
import gymnasium as gym

print(sinergym.__version__)
print(sinergym.ids())

# Make and consult environment
env = gym.make('Eplus-5zone-hot-continuous-stochastic-v1')
print(env.get_wrapper_attr('to_str')())

