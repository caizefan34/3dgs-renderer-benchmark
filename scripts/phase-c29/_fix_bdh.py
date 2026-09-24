for fn in ['b_param_frequency.py','d_densification_frequency.py','h_event_prediction.py']:
    c=open('scripts/phase-c29/'+fn).read()
    c=c.replace('loss=(rc-gt).abs().mean()+0.2*(1.0-min(1.0,1.0))\n    loss.backward()','loss=(rc-gt).abs().mean()+0.2*(1.0-min(1.0,1.0)); loss.backward()')
    open('scripts/phase-c29/'+fn,'w').write(c)
    print('fixed',fn)
