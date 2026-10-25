import cv2, numpy as np, sys; sys.path.insert(0,'.')
from tipcount.monitor import SlotMonitor, WARP_W, WARP_H, RACK_COLOURS, DST, pixel_classes
np.set_printoptions(precision=2, suppress=True, linewidth=200)
mon=SlotMonitor('tipcount/slots.json')
for spec_s in sys.argv[1:]:
    f,slot,colour=spec_s.split(':'); spec=RACK_COLOURS[colour]
    im=cv2.imread(f); hsv=cv2.cvtColor(im,cv2.COLOR_BGR2HSV)
    fr=mon.read(im); q=mon.slots[slot]['quad']+mon.track_state[slot]['offset']
    M=cv2.getPerspectiveTransform(q,DST)
    warp=cv2.warpPerspective(hsv,M,(WARP_W,WARP_H),flags=cv2.INTER_LINEAR)
    pf,rf,ff,ring=mon._classify(warp,colour)
    tips=(pf>=0.5)&(ring<=0.10) if colour=='yellow' else pf>=spec['tip_pf']
    print(f'===== {f} {slot} {colour} tips={int(tips.sum())} offset={mon.track_state[slot]["offset"].tolist()}')
    print('pale centre\n',pf); print('ring rim\n',ring); print('frame\n',ff)
    w=cv2.warpPerspective(im,M,(WARP_W,WARP_H)); w=cv2.resize(w,None,fx=6,fy=6,interpolation=cv2.INTER_CUBIC)
    for i in range(1,12): cv2.line(w,(i*48,0),(i*48,384),(0,0,255),1)
    for i in range(1,8): cv2.line(w,(0,i*48),(576,i*48),(0,0,255),1)
    for r in range(8):
        for c in range(12): cv2.circle(w,(c*48+24,r*48+24),4,(0,255,0) if tips[r,c] else (0,0,255),-1)
    cv2.imwrite(f'scratch/sc3_{f[:-4]}_{slot}.png',w)
