import torch
import random
import numpy as np



info = {'name':'班长', 'id':100, 'sex':'f', 'address':'北京'}
age = info.get('age')
print(age)
age=info.get('age',18)
print(info)

