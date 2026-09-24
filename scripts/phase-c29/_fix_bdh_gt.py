for fn in ['b_param_frequency.py','d_densification_frequency.py','h_event_prediction.py']:
    c=open('scripts/phase-c29/'+fn).read()
    c=c.replace('.unsqueeze(0).permute(0,3,1,2).contiguous(); del m','.detach().unsqueeze(0).permute(0,3,1,2).contiguous(); del m')
    open('scripts/phase-c29/'+fn,'w').write(c)
    print('fixed gt.detach() in',fn)
