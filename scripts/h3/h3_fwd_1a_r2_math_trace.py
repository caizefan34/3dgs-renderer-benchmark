#!/usr/bin/env python3
import json, struct, numpy as np
P=[(1511.31201171875,115.58989715576172,.06679920852184296,-.06483334302902222,.15477153658866882,.9991458654403687,95,6,11,1),(390.7921447753906,1023.1334228515625,.6001740097999573,.31892746686935425,.21786050498485565,.999340832233429,24,62,3,15)]
def bit(x):x=np.float32(x);return {'value':float(x),'hex':'0x%08x'%struct.unpack('<I',struct.pack('<f',x))[0]}
def one(q):
 x,y,A,B,C,o,tx,ty,mx,my=map(np.float32,q);disc=B*B-A*C;t=np.minimum(np.float32(4096*4096),np.float32(2)*np.log(o/np.float32(1/255),dtype=np.float32));n=-t/disc;ex=np.sqrt(n*C,dtype=np.float32);ey=np.sqrt(n*A,dtype=np.float32);bminx=x-ex;bmaxx=x+ex;bminy=y-ey;bmaxy=y+ey;amin_y=y+B*ex/C;amax_y=y-B*ex/C;amin_x=x+B*ey/A;amax_x=x-B*ey/A;rx0=int(bminx/16);rx1=int(bmaxx/16+1);ry0=int(bminy/16);ry1=int(bmaxy/16+1);isY=(ry1-ry0)<(rx1-rx0)
 x0,x1=tx*16,(tx+1)*16;y0,y1=ty*16,(ty+1)*16;qs=[]
 for xx in (x0,x1):
  yy=np.minimum(np.maximum(y-B*(xx-x)/C,y0),y1);qs.append(A*(xx-x)*(xx-x)+2*B*(xx-x)*(yy-y)+C*(yy-y)*(yy-y))
 for yy in (y0,y1):
  xx=np.minimum(np.maximum(x-B*(yy-y)/A,x0),x1);qs.append(A*(xx-x)*(xx-x)+2*B*(xx-x)*(yy-y)+C*(yy-y)*(yy-y))
 qmin=min(qs)
 return {'disc':bit(disc),'t':bit(t),'neg_t_over_disc':bit(n),'x_extent':bit(ex),'y_extent':bit(ey),'bbox_min':[bit(bminx),bit(bminy)],'bbox_max':[bit(bmaxx),bit(bmaxy)],'bbox_argmin':[bit(amin_y),bit(amin_x)],'bbox_argmax':[bit(amax_y),bit(amax_x)],'rect_min':[rx0,ry0],'rect_max':[rx1,ry1],'isY':bool(isY),'old_hit_rect_qmin':bit(qmin),'old_hit_rect_includes':bool(qmin<=t),'target_tile':[int(tx),int(ty)],'target_macro':[int(mx),int(my)]}
print(json.dumps([one(x) for x in P],indent=2))
