x = [1,1,1,1]
v = [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
index = [0,0,0,0]

def func():
    for t in range(4):
        if x[t] >= v[t]:
            index[t] = 1
        else:
            index[t] = 0
    print(index)

def myFunc():
    for step in range(4):


func()