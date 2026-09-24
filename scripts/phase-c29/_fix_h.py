c=open('scripts/phase-c29/h_event_prediction.py').read()
c=c.replace(').abs().mean()+0.2*(1.0-min(1.0,1.0)).backward()',').abs().mean()+0.2*(1.0-min(1.0,1.0))\n    loss.backward()')
open('scripts/phase-c29/h_event_prediction.py','w').write(c)
print('fixed h')
