file_his = open('Historyaction.txt', 'r', encoding='UTF-8') 
line_his = file_his.readlines()
a = line_his[2]
a = a.strip('[').strip('\n').strip(']').split(', ')
for i in range(len(a)):
    a[i] = int(a[i])
    print(a[i])
