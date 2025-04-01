from ctypes import *
import lzma
p='/kaggle_simulations/agent/'
z=lzma.open(p+'O.so.xz')
y=z.read()
z.close()
x=open(p+'O.so','wb')
x.write(y)
x.close()
del y
a=CDLL(p+'O.so')
a.go.restype=c_char_p
a.init()
r=None
m=''
def ag(o,c):
 global r,m
 if r is None:
  r=o['board']
 else:
  m+=o['lastMove']+' '
 wt=int(o['remainingOverageTime']*1000)
 bt=int(o['opponentRemainingOverageTime']*1000)
 if o['mark']!='white':
  wt,bt=bt,wt
 wt+=70
 bt+=70
 a.position(r.encode(),m.encode())
 ret=a.go(wt,bt,0,0).decode()
 m+=ret+' '
 return ret
 
