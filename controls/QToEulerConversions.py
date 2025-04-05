import math
#https://en.wikipedia.org/wiki/Conversion_between_quaternions_and_Euler_angles
def eToQ(angles):
    x,y,z=[a/2 for a in angles] #dividing by 2 cause all of the trigs need it
    Q=[0]*4
    cr=math.cos(x)
    sr=math.sin(x)
    cp=math.cos(y)
    sp=math.sin(y)
    cy=math.cos(z)
    sy=math.sin(z)
    Q[0]=cr*cp*cy+sr*sp*sy
    Q[1]=sr*cp*cy-cr*sp*sy
    Q[2]=cr*sp*cy+sr*cp*sy
    Q[3]=cr*cp*sy-sr*sp*cy
    #comes out normalized
    return Q

def qToE(quaternion):
    a,b,c,d=quaternion
    angles=[0]*3
    angles[0]=math.atan2(2*(a*b+c*d),1-2*(b*b+c*c))
    #angles[1]=-math.pi/2+2*math.atan2(math.sqrt(1+2*(a*c-b*d)),math.sqrt(1-2*(a*c-b*d)))
    angles[1]=math.asin(2*(a*c-b*d))
    angles[2]=math.atan2(2*(a*d+b*c),1-2*(c*c+d*d))
    return angles
    
angles=[1,-1.2,3] #in radians
q=eToQ(angles)
print(qToE(q))

