from __future__ import print_function
s = open('__CmainBody.txt', 'r').read()
fmt = [("ct", "Cos[theta]"),
        ("exp", "E^"),
        ("r0", "r^0"),
        ("r1", "r^1"),
        ("r2", "r^2"),
        ("r3", "r^3"),
        ("r4", "r^4"),
        ("r5", "r^5"),
        ("p", "phi"),
        ("p2", "p^2"),
        ("pa2", "pa^2")
]
fmt.extend([("r{0}".format(r), "r^{0}".format(r)) for r in range(10)])
fmt.extend([("ct{0}".format(r), "ct^{0}".format(r)) for r in range(10)])

for new, old in fmt:
    s = s.replace(old, new)
    
reps = [
    ('', " "),
    ("ct" , "cos(theta)"),
    ("r2" , "r**2"),
    ("r3" , "r**3"),
    ("r4" , "r2**2"),
    ("ct2" , "ct**2"),
    ("ct3" , "ct**3"),
    ("ct4" , "ct2**2"),
    ("ct5" , "ct**5"),
    ("ct6" , "ct2**3"),
    ("p2" , "phi**2"),
    ("H" , "#1"),
    ("poly " , "2*pa*p2*r5+2*pa*p2*r5*ct-2*pa^2*p2*r5*ct-2*pa*p2*r5*ct2-2*pa*p2*r5*ct3+2*pa^2*p2*r5*ct3+(r4+4*p2*r4+4*pa*p2*r4+r4*ct-2*pa*r4*ct+4*p2*r4*ct-2*pa*p2*r4*ct-2*pa^2*p2*r4*ct-r4*ct2-4*p2*r4*ct2-r4*ct3+2*pa*r4*ct3-4*p2*r4*ct3+6*pa*p2*r4*ct3-2*pa^2*p2*r4*ct3)*H+(4*r3+8*p2*r3+2*pa*p2*r3+3*r3*ct-6*pa*r3*ct+4*p2*r3*ct-4*pa*p2*r3*ct-3*r3*ct2-4*p2*r3*ct2+2*pa*p2*r3*ct2-2*r3*ct3+4*pa*r3*ct3)*H^2+(6*r2+4*p2*r2+3*r2*ct-6*pa*r2*ct-3*r2*ct2-r2*ct3+2*pa*r2*ct3)*H^3+(4*r+r*ct-2*pa*r*ct-r*ct2)*H^4+H^5"),
    ("rt1" , "Root[poly &,1]"),
    ("rt2" , "Root[poly &,2]"),
    ("rt3" , "Root[poly &,3]"),
    ("rt4" , "Root[poly &,4]"),
    ("rt5" , "Root[poly &,5]"),
    ("etr1", "exp(t*rt1)"),
    ("etr2", "exp(t*rt2)"),
    ("etr3", "exp(t*rt3)"),
    ("etr4", "exp(t*rt4)"),
    ("etr5", "exp(t*rt5)"),
]
for p in range(2, 5):
    reps.extend([("rt{}p{}".format(r, p), "rt{}^{}".format(r,p)) for r in range(1,6)])

for new, old in reps:
    s = s.replace(old, new)

s = s.replace('/((rt1-rt2)*(rt1-rt3)*(rt2-rt3)*(rt1-rt4)*(rt2-rt4)*(rt3-rt4)*(rt1-rt5)*(rt2-rt5)*(rt3-rt5)*(rt4-rt5))\n', '')

with open('__CmainBody.txt', 'w') as f:
     print(s, file=f)

# pattern recog
sr = s
for r in range(5):
    sr = sr.replace('etr{}'.format(r), '')
    
sr = sr.replace('-', '+')
sr = sr.replace(' ', '')
sr = sr.split('+')

sr = map(lambda x: x[1:] if x[0] == '*' else x, sr)
sr = map(lambda x: x[3:] if x[1:3] == '**' else x, sr)
sr = map(lambda x: x[3:] if x[0:3] == 'pa*' else x, sr)
sr = map(lambda x: x[5:] if x[0:5] == 'pa^2*' else x, sr)




root = "2*pa*p2*r5+2*pa*p2*r5*ct-2*pa^2*p2*r5*ct-2*pa*p2*r5*ct2-2*pa*p2*r5*ct3+2*pa^2*p2*r5*ct3+(r4+4*p2*r4+4*pa*p2*r4+r4*ct-2*pa*r4*ct+4*p2*r4*ct-2*pa*p2*r4*ct-2*pa^2*p2*r4*ct-r4*ct2-4*p2*r4*ct2-r4*ct3+2*pa*r4*ct3-4*p2*r4*ct3+6*pa*p2*r4*ct3-2*pa^2*p2*r4*ct3)*H+(4*r3+8*p2*r3+2*pa*p2*r3+3*r3*ct-6*pa*r3*ct+4*p2*r3*ct-4*pa*p2*r3*ct-3*r3*ct2-4*p2*r3*ct2+2*pa*p2*r3*ct2-2*r3*ct3+4*pa*r3*ct3)*H^2+(6*r2+4*p2*r2+3*r2*ct-6*pa*r2*ct-3*r2*ct2-r2*ct3+2*pa*r2*ct3)*H^3+(4*r+r*ct-2*pa*r*ct-r*ct2)*H^4+H^5".split("H")

root[1] = root[1][1:]
for i in range(1,6):
    root[i] = root[i][3:]
root[4] = "1"
root.pop(5)
root.reverse()
root.append('0')
    


from sympy import Symbol, Poly, cos

def fn(theta, phi, pa, tau):
    x = Symbol('x')
    pa = Symbol('pa')
    phi = Symbol('phi')
    theta = Symbol('theta')
    tau = Symbol('tau')
    f = (2*pa*phi**2)/tau**5 + x**5 + (2*pa*phi**2*cos(theta))/tau**5 - (2*pa**2*phi**2*cos(theta))/tau**5 - (2*pa*phi**2*cos(theta)**2)/tau**5 - (2*pa*phi**2*cos(theta)**3)/tau**5 + (2*pa**2*phi**2*cos(theta)**3)/tau**5 + x**4*(4/tau + cos(theta)/tau - (2*pa*cos(theta))/tau - cos(theta)**2/tau) + x*(tau**(-4) + (4*phi**2)/tau**4 + (4*pa*phi**2)/tau**4 + cos(theta)/tau**4 - (2*pa*cos(theta))/tau**4 + (4*phi**2*cos(theta))/tau**4 - (2*pa*phi**2*cos(theta))/tau**4 - (2*pa**2*phi**2*cos(theta))/tau**4 - cos(theta)**2/tau**4 - (4*phi**2*cos(theta)**2)/tau**4 - cos(theta)**3/tau**4 + (2*pa*cos(theta)**3)/tau**4 - (4*phi**2*cos(theta)**3)/tau**4 + (6*pa*phi**2*cos(theta)**3)/tau**4 - (2*pa**2*phi**2*cos(theta)**3)/tau**4) + x**2*(4/tau**3 + (8*phi**2)/tau**3 + (2*pa*phi**2)/tau**3 + (3*cos(theta))/tau**3 - (6*pa*cos(theta))/tau**3 + (4*phi**2*cos(theta))/tau**3 - (4*pa*phi**2*cos(theta))/tau**3 - (3*cos(theta)**2)/tau**3 - (4*phi**2*cos(theta)**2)/tau**3 + (2*pa*phi**2*cos(theta)**2)/tau**3 - (2*cos(theta)**3)/tau**3 + (4*pa*cos(theta)**3)/tau**3) + x**3*(6/tau**2 + (4*phi**2)/tau**2 + (3*cos(theta))/tau**2 - (6*pa*cos(theta))/tau**2 - (3*cos(theta)**2)/tau**2 - cos(theta)**3/tau**2 + (2*pa*cos(theta)**3)/tau**2)
    
    p = Poly(f,x)
    ans = [p.root(k).evalf() for k in range(p.degree())]
    for i,a in enumerate(ans):
        print(i, a)
    return ans