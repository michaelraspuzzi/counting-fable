import cv2, numpy as np, sys; sys.path.insert(0,'.')
from tipcount.monitor import SlotMonitor, RACK_COLOURS, WARP_W, WARP_H, ROWS, COLS, CELL
np.set_printoptions(precision=2, suppress=True, linewidth=200)
mon=SlotMonitor('tipcount/slots.json')
for spec_s in sys.argv[1:]:
    f,slot,colour=spec_s.split(':'); spec=RACK_COLOURS[colour]
    im=cv2.imread(f); hsv=cv2.cvtColor(im,cv2.COLOR_BGR2HSV); sl=mon.slots[slot]
    w=cv2.warpPerspective(hsv,sl['M'],(WARP_W,WARP_H),flags=cv2.INTER_LINEAR)
    h,s,v=w[:,:,0],w[:,:,1],w[:,:,2]
    hue_ok=(h>=spec['h'][0])&(h<=spec['h'][1]); sat=hue_ok&(s>=spec['hole_s'])
    lat=np.percentile(v[sat],95) if sat.sum()>20 else 255
    pale=hue_ok&(s>=45)&(s<spec['pale_s'])&(v>spec['pale_v'])
    rim=sat&(v<spec['hole_v_ratio']*lat)&(v>=spec['hole_v_min'])
    P=pale.reshape(ROWS,CELL,COLS,CELL); R=rim.reshape(ROWS,CELL,COLS,CELL)
    pc=P[:,2:6,:,2:6].mean(axis=(1,3))
    ring=R.sum(axis=(1,3))-R[:,2:6,:,2:6].sum(axis=(1,3)); ring=ring/(64-16)
    print('=====',f,slot,colour); print('pale centre 4x4\n',pc); print('rim ring\n',ring)
