import cv2, numpy as np, sys; sys.path.insert(0,'.')
from tipcount.monitor import SlotMonitor, WARP_W, WARP_H, RACK_COLOURS
np.set_printoptions(precision=2, suppress=True, linewidth=200)
mon=SlotMonitor('tipcount/slots.json')
for spec_s in sys.argv[1:]:
    f,slot,colour=spec_s.split(':'); spec=RACK_COLOURS[colour]
    im=cv2.imread(f); hsv=cv2.cvtColor(im,cv2.COLOR_BGR2HSV); sl=mon.slots[slot]
    warp=cv2.warpPerspective(hsv,sl['M'],(WARP_W,WARP_H),flags=cv2.INTER_LINEAR)
    pf,rf,ff,ring=mon._classify(warp,colour)
    h,s,v=warp[:,:,0],warp[:,:,1],warp[:,:,2]
    hue_ok=(h>=spec['h'][0])&(h<=spec['h'][1]); sat=hue_ok&(s>=spec['hole_s'])
    lat=np.percentile(v[sat],95) if sat.sum()>20 else -1
    tips=(pf>=0.5)&(ring<=0.10) if colour=='yellow' else pf>=0.4
    print(f'===== {f} {slot} {colour} tips={int(tips.sum())} lattice_v={lat:.0f} frameS_med={np.median(s[sat]) if sat.sum() else -1:.0f} pale_px={int(((hue_ok)&(s>=45)&(s<spec["pale_s"])&(v>spec["pale_v"])).sum())} medV_all={np.median(v):.0f} medS_all={np.median(s):.0f}')
    print('pale centre\n',pf); print('ring rim\n',ring); print('frame\n',ff)
    w=cv2.warpPerspective(im,sl['M'],(WARP_W,WARP_H)); w=cv2.resize(w,None,fx=6,fy=6,interpolation=cv2.INTER_CUBIC)
    for i in range(1,12): cv2.line(w,(i*48,0),(i*48,384),(0,0,255),1)
    for i in range(1,8): cv2.line(w,(0,i*48),(576,i*48),(0,0,255),1)
    for r in range(8):
        for c in range(12): cv2.circle(w,(c*48+24,r*48+24),4,(0,255,0) if tips[r,c] else (0,0,255),-1)
    cv2.imwrite(f'scratch/sc2_{f[:-4]}_{slot}.png',w)
