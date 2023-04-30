import os
path = r'D:\my_job\DATA\INRIAPerson\inriaPerson\train'

file = os.listdir(path)
file = [i  for i in file if i.endswith('.png')]
print(file)