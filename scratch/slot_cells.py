import cv2, numpy as np, sys; sys.path.insert(0,'.')
from tipcount.monitor import SlotMonitor, DST, WARP_W, WARP_H
np.set_printoptions(precision=2, suppress=True, linewidth=200)
mon=SlotMonitor('tipcount/slots.json')
for spec in sys.argv[1:]:
    f,slot,colour=spec.split(':')
    im=cv2.imread(f); hsv=cv2.cvtColor(im,cv2.COLOR_BGR2HSV); sl=mon.slots[slot]
    warp=cv2.warpPerspective(hsv,sl['M'],(WARP_W,WARP_H),flags=cv2.INTER_LINEAR)
    pf,rf,ff=mon._classify(warp,colour)
    print(f'===== {f} {slot} {colour}  tips={int(((pf>=0.5)&(pf-rf>=0.35)).sum())}')
    print('pale\n',pf); print('rim\n',rf); print('frame\n',ff)
    w=cv2.warpPerspective(im,sl['M'],(WARP_W,WARP_H)); w=cv2.resize(w,None,fx=6,fy=6,interpolation=cv2.INTER_CUBIC)
    for i in range(1,12): cv2.line(w,(i*48,0),(i*48,384),(0,0,255),1)
    for i in range(1,8): cv2.line(w,(0,i*48),(576,i*48),(0,0,255),1)
    for r in range(8):
        for c in range(12):
            tip=(pf[r,c]>=0.5)&(pf[r,c]-rf[r,c]>=0.35)
            cv2.circle(w,(c*48+24,r*48+24),4,(0,255,0) if tip else (0,0,255),-1)
    cv2.imwrite(f'scratch/sc_{f[:-4]}_{slot}.png',w)
