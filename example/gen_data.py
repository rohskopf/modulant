import numpy as np

def function(x):
    value = 0.
    for i in range(0,ninputs):
        value+=x[i]**2
    return value

nsamples = 4
ninputs = 2
noutputs = 1

fh_in = open('TIN', 'w')
fh_out = open('TOUT', 'w')
for d in range(0,nsamples):
    x = np.random.uniform(low=-1., high=1., size=ninputs)
    #print(x)
    value = function(x)
    #print('value: %f' % (value))
    for i in range(0,ninputs):
        fh_in.write('%f ' % (x[i]))
    fh_in.write('\n')
    fh_out.write('%f\n' % (value))

fh_in.close()
fh_out.close()
    
