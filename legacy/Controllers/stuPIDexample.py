#I'm so funny, am I not?
import matplotlib.pyplot as mp
import math
def error(curr,target): #just a linear error function. Make sure that your error has direction (i.e. can be negative or positive. I think linear error might be simplest for what we are doing but other error funcs may work better who knows, certainly not me)
    return target-curr
def f(x): #arbitrary test function
    return math.cos(x)
lwe=1 #linear weight
iwe=.01 #integral weight
dwe=-.01 #derivative weight

integral=0
prev=0
curr=0
dt=.2

y=[]
mew=[]
x=[]
for i in range(1000):
    mew.append(f(i*dt))
    err=error(curr,mew[-1])
    integral+=err*dt
    der=(err-prev)/dt
    curr+=err*lwe+integral*iwe+der*dwe
    prev=err
    #print(curr,err)
    y.append(curr)
    x.append(i*dt)
mp.plot(x,y) #blue line is pid output
mp.plot(x,mew) #orange line is function (desired) output
mp.show()

