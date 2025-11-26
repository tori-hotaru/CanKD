import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmrazor.registry import MODELS
from mmengine.runner.checkpoint import _load_checkpoint

import cv2
from time import time
import numpy as np

data_format = 'NHWC'

class DepthToSpace(torch.nn.Module):

    def __init__(self, h_factor=2, w_factor=2):
        super().__init__()
        self.h_factor, self.w_factor = h_factor, w_factor
    
    def forward(self, x):
        return pixelShuffle(x, self.h_factor, self.w_factor)

class SpaceToDepth(torch.nn.Module):

    def __init__(self, h_factor=2, w_factor=2):
        super().__init__()
        self.h_factor, self.w_factor = h_factor, w_factor
    
    def forward(self, x):
        return inv_pixelShuffle(x, self.h_factor, self.w_factor)

class ContourDec(torch.nn.Module):

    def __init__(self, nlevs):
        super().__init__()
        self.nlevs = nlevs
    
    def forward(self, x):
        return pdfbdec_layer(x, self.nlevs)

class ContourRec(torch.nn.Module):

    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return pdfbrec_layer(x[0], x[1])

def dup(x, step=[2,2]):
    N,C,H,W = x.shape
    y = torch.zeros((N,C,H*step[0],W*step[1]), device=x.device)
    y[...,::step[0],::step[1]]=x
    return y
    
def conv2(x, W, C=1, strides=[1, 1, 1, 1], padding=0):
    return F.conv2d(x, W, padding=padding, groups=C)


def extend2_layer(x, ru, rd, cl, cr, extmod):
    # rx, cx = x.get_shape().as_list()[1:3]
    rx, cx = x.shape[2:]
    if extmod == 'per':

        y = torch.cat([x[..., rx-ru:rx,:],x,x[..., :rd,:]], dim=2)
        y = torch.cat([y[..., cx-cl:cx],y,y[..., :cr]], dim=3)

    elif extmod == 'qper_row':
        raise ValueError
        rx2 = round(rx / 2)
        y1 = K.concatenate([x[:,rx2:rx, cx-cl:cx,:], x[:,:rx2, cx-cl:cx,:]],axis=1)
        y2=K.concatenate([x[:,rx2:rx, :cr,:], x[:,:rx2, :cr,:]],axis=1)
        y=K.concatenate([y1,x,y2], axis=1)
        
        y=K.concatenate([y[:,rx-ru:rx,:,:],y,y[:,:rd,:,:]], axis=1)

    elif extmod == 'qper_col':
        
        cx2 = round(cx / 2)
        y1 = torch.cat([x[..., rx-ru:rx, cx2:cx], x[..., rx-ru:rx, :cx2]],dim=3)
        y2 = torch.cat([x[..., :rd, cx2:cx], x[..., :rd, :cx2]],dim=3)
        y = torch.cat([y1,x,y2], dim=2)
        
        y = torch.cat([y[..., cx-cl:cx],y,y[..., :cr]],dim=3)
    return y

def sefilter2_layer(x, f1, f2, extmod='per',shift=[0,0]):
    # Periodized extension
    f1_len = len(f1)
    f2_len = len(f2)
    lf1 = (f1_len - 1) / 2
    lf2 = (f2_len - 1) / 2
    
    y = extend2_layer(x, int(np.floor(lf1) + shift[0]), int(np.ceil(lf1) - shift[0]), \
         int(np.floor(lf2) + shift[1]), int(np.ceil(lf2) - shift[1]),extmod)

    # Seperable filter
    ch = y.shape[1]
    # f3=np.zeros((f1_len, 1, ch, ch), dtype = np.float32)
    f3 = torch.zeros((ch, ch, f1_len, 1), device=x.device)
    for i in range(ch):
        # f3[:,0,i,i] = f1
        f3[i,i,:,0] = f1
        
    # f4=np.zeros((1, f2_len, ch, ch), dtype = np.float32)
    f4 = torch.zeros((ch, ch, 1, f2_len), device=x.device)
    for i in range(ch):
        # f4[0,:,i,i] = f2
        f4[i,i,0,:] = f2
    
    y = conv2(y, f3)
    y = conv2(y, f4)
    
    return y

def lap_filter(device, dtype):
    # use '9-7' filter for the Laplacian pyramid
    h = np.array([0.037828455506995, -0.02384946501938, -0.11062440441842, 0.37740285561265], dtype = np.float32)
    h = np.concatenate((h,[0.8526986790094],h[::-1]))
        
    g = np.array([-0.064538882628938, -0.040689417609558, 0.41809227322221], dtype = np.float32)
    g = np.concatenate((g,[0.78848561640566],g[::-1]))
    h, g = torch.from_numpy(h), torch.from_numpy(g)
    h, g = h.to(device), g.to(device)
    return h, g

def lpdec_layer(x):
    h, g = lap_filter(x.device, x.dtype)

    # Lowpass filter and downsample
    xlo = sefilter2_layer(x,h,h,'per')
    c = xlo[:,:,::2,::2]

    # Compute the residual (bandpass) image by upsample, filter, and subtract
    # Even size filter needs to be adjusted to obtain perfect reconstruction
    adjust = (len(g) + 1)%2
    
    # d = insert_zero(xlo) # d = Lambda(insert_zero)(xlo)
    d = dup(c) # d = Lambda(insert_zero)(xlo)

    d = sefilter2_layer(d,g,g,'per',adjust*np.array([1,1], dtype = np.float32))
    
    d = x-d # d = Subtract()([x,d])

    return c,d

def lprec_layer(c,d):
    h, g = lap_filter(c.device, c.dtype)

    xhi = sefilter2_layer(d,h,h,'per')
    xhi = xhi[...,::2,::2] # xhi = Lambda(lambda x: x[:,::2,::2,:])(xhi)

    xlo = c - xhi # xlo = Subtract()([c, xhi])
    xlo = dup(xlo) # xlo = Lambda(dup)(xlo)
    
    # Even size filter needs to be adjusted to obtain 
    # perfect reconstruction with zero shift
    adjust = (len(g) + 1)%2
    
    xlo = sefilter2_layer(xlo,g,g,'per',adjust*np.array([1,1]))
    
    # Final combination
    x = xlo + d # x = Add()([xlo, d])
    
    return x

def pdfbdec_layer(x, nlevs, pfilt=None, dfilt=None):
    if nlevs != 0:
        # Laplacian decomposition
        xlo, xhi = lpdec_layer(x)

        # Use the ladder structure (whihc is much more efficient)
        xhi = dfbdec_layer(xhi, dfilt, nlevs)

    return xlo, xhi

def dfb_filter(device):
    # length 12 filter from Phoong, Kim, Vaidyanathan and Ansari
    v = np.array([0.63,-0.193,0.0972,-0.0526,0.0272,-0.0144], dtype = np.float32)
    # Symmetric impulse response
    f = np.concatenate((v[::-1],v))
    # Modulate f
    f[::2] = -f[::2]
    f = torch.from_numpy(f)
    f = f.to(device)
    return f

def new_fbdec_layer(x, f_, type1, type2, extmod='per'):

    sample = len(x)
    # x = tf.concat(x, axis=-1)
    x = torch.cat(x, dim=1)
    ch = x.shape[1] // sample

    # Polyphase decomposition of the input image
    if type1 == 'q':
        # Quincunx polyphase decomposition
        p0,p1 = qpdec_layer(x,type2)
        
    elif type1 == 'p':
        # Parallelogram polyphase decomposition
        p0,p1 = ppdec_layer(x,type2)

    # Ladder network structure
    y0 = 1 / (2**0.5) * (p0 - sefilter2_layer(p1,f_,f_,extmod,[1,1]))
    y1 = (-2**0.5)*p1 - sefilter2_layer(y0,f_,f_,extmod)
    
    # return [y0, y1]
    return [y0[:,i*ch:(i+1)*ch] for i in range(sample)], [y1[:,i*ch:(i+1)*ch] for i in range(sample)]

### TODO DFB
def dfbdec_layer(x, f, n):
    f = dfb_filter(x.device)

    if n == 1:
        y = [None] * 2
        # Simplest case, one level
        y[0], y[1] = fbdec_layer(x, f, 'q', '1r', 'qper_col')
    elif n >= 2:
        y = [None] * 4
        x0, x1 = fbdec_layer(x, f, 'q', '1r', 'qper_col')
        # y[1], y[0] = fbdec_layer(x0, f, 'q', '2c', 'per')
        # y[3], y[2] = fbdec_layer(x1, f, 'q', '2c', 'per')
        odd_list, even_list = new_fbdec_layer([x0, x1], f, 'q', '2c', 'per')
        # y[1], y[2] = odd_list
        # y[0], y[3] = even_list
        for ix in range(len(odd_list)):
            y[ix*2+1], y[ix*2] = odd_list[ix], even_list[ix]
        
        # Now expand the rest of the tree
        for l in range(3,n+1):
            # Allocate space for the new subband outputs
            y_old = y.copy()
            y = [None] * (2**l)
            
            # The first half channels use R1 and R2
            # for k in range(1, 2 ** (l - 2)+1):
            #     i = (k - 1) % 2 + 1
            #     y[2*k-1], y[2*k-2] = fbdec_layer(y_old[k-1], f, 'p', i, 'per')
            odd = np.arange(1, 2 ** (l - 2)+1, 2)
            even = np.arange(2, 2 ** (l - 2)+1, 2)
            odd_list, even_list = new_fbdec_layer([y_old[k-1] for k in odd], f, 'p', 1, 'per')
            for ix, k in enumerate(odd):
                y[2*k-1], y[2*k-2] = odd_list[ix], even_list[ix]
            odd_list, even_list = new_fbdec_layer([y_old[k-1] for k in even], f, 'p', 2, 'per')
            for ix, k in enumerate(even):
                y[2*k-1], y[2*k-2] = odd_list[ix], even_list[ix]
                
            # The second half channels use R3 and R4
            # for k in range(2 ** (l - 2) + 1,2 ** (l - 1) + 1):
            #     i = (k - 1) % 2 + 3
            #     y[2*k-1], y[2*k-2] = fbdec_layer(y_old[k-1], f, 'p', i, 'per')
            odd += 2 ** (l - 2)
            even += 2 ** (l - 2)
            odd_list, even_list = new_fbdec_layer([y_old[k-1] for k in odd], f, 'p', 3, 'per')
            for ix, k in enumerate(odd):
                y[2*k-1], y[2*k-2] = odd_list[ix], even_list[ix]
            odd_list, even_list = new_fbdec_layer([y_old[k-1] for k in even], f, 'p', 4, 'per')
            for ix, k in enumerate(even):
                y[2*k-1], y[2*k-2] = odd_list[ix], even_list[ix]
    
    # Backsampling
    def backsamp(y=None):
        n = np.log2(len(y))
        
        assert not (n != round(n) or n < 1), 'Input must be a cell vector of dyadic length'
        n=int(n)
        if n == 1:
            # One level, the decomposition filterbank shoud be Q1r
            # Undo the last resampling (Q1r = R2 * D1 * R3)
            for k in range(2):
                y[k]=resamp(y[k],4)
                y[k][..., ::2]=resamp(y[k][..., ::2], 1)
                y[k][..., 1::2]=resamp(y[k][..., 1::2],1)
        
        if n > 2:
            N=2 ** (n - 1)
                
            for k in range(1, 2 ** (n - 2) +1):
                shift = 2 * k - (2 ** (n - 2) + 1)
                
                # The first half channels
                # y[2*k - 2]=resamp(y[2*k - 2],3,shift)
                # y[2*k - 1]=resamp(y[2*k - 1],3,shift)
                y[2*k - 2], y[2*k - 1] = new_resamp([y[2*k - 2], y[2*k - 1]],3,shift)
                
                # The second half channels
                # y[2*k - 2 + N]=resamp(y[2*k - 2 + N],1,shift)
                # y[2*k - 1 + N]=resamp(y[2*k - 1 + N],1,shift)
                y[2*k - 2 + N], y[2*k - 1 + N] = new_resamp([y[2*k - 2 + N], y[2*k - 1 + N]],1,shift)

        return y
    y=backsamp(y)
    
    # Flip the order of the second half channels
    y[2 ** (n - 1):]=y[-1:2 ** (n - 1)-1:-1]

    return y

### TODO DFB
def new_resamp(y, type_, shift=1):
    sample = len(y)
    # print(y[0].shape)
    # print(y[1].shape)
    y = torch.stack(y)
    # print(y.get_shape().as_list())
    if type_ in [3,4]:
        y = torch.transpose(y, 3, 4)

    # m,n,c=y.get_shape().as_list()[-3:]
    N,c,m,n = y.shape[1:]

    # y = tf.reshape(y, [sample, -1, m*n, c])
    y = torch.reshape(y, [sample, -1, c, m*n])

    z=np.zeros((m,n), dtype=np.int64)
    for j in range(n):
        if type_ in [1,3]:
            k= (shift * j) % m
            
        else:
            k= (-shift * j) % m
            
        if k < 0:
            k=k + m
        t1 = np.arange(k, m)
        t2 = np.arange(k)
        z[:,j] = np.concatenate([t1, t2]) * n + j
        
    z = z.reshape(-1)
    z = torch.from_numpy(z) # LongTensor int64
    z = z.to(y.device)

    z = torch.reshape(z, (1,1,1,-1))
    y = torch.gather(y, 3, z.expand(sample,N,c,-1))

    # y = tf.gather(y, z.astype(int), axis=-2)
    # y = Reshape((m,n,c))(y)
    # y = tf.reshape(y, [sample, -1, m, n, c])
    y = torch.reshape(y, [sample, -1, c, m, n])

    if type_ in [3,4]:
        y = torch.transpose(y, 3, 4)
    y = [y[i] for i in range(sample)]
    return y

def fbdec_layer(x, f_, type1, type2, extmod='per'):

    # Polyphase decomposition of the input image
    if type1 == 'q':
        # Quincunx polyphase decomposition
        p0,p1 = qpdec_layer(x,type2)
        
    elif type1 == 'p':
        # Parallelogram polyphase decomposition
        p0,p1 = ppdec_layer(x,type2)
    
    # Ladder network structure
    y0 = 1 / (2**0.5) * (p0 - sefilter2_layer(p1,f_,f_,extmod,[1,1]))
    y1 = (-2**0.5)*p1 - sefilter2_layer(y0,f_,f_,extmod)
    
    return [y0, y1]

def qpdec_layer(x, type_='1r'):

    if type_ == '1r':   # Q1 = R2 * D1 * R3
        y = resamp(x, 2)

        # p0 = resamp(y[:,::2,:,:], 3)
        
        # inv(R2) * [0; 1] = [1; 1]
        # p1 = resamp(y(2:2:end, [2:end, 1]), 3)
        p1 = torch.cat([y[..., 1::2,1:], y[..., 1::2,0:1]], dim=3)
        # p1 = resamp(p1, 3)
        p0, p1 = new_resamp([y[...,::2,:], p1], 3)

    elif type_ == '1c': # Q1 = R3 * D2 * R2
        # TODO
        y=resamp(x,3)
        
        # p0=resamp(y[:,:,::2,:],2)
        p0=resamp(y[...,::2],2)
        
        # inv(R3) * [0; 1] = [0; 1]
        # p1=resamp(y[:,:,1::2,:],2)
        p1=resamp(y[...,1::2],2)
            
    elif type_ == '2r': # Q2 = R1 * D1 * R4
        # TODO
        y=resamp(x,1)
        
        p0=resamp(y[...,::2,:],4)
        
        # inv(R1) * [1; 0] = [1; 0]
        p1=resamp(y[...,1::2,:],4)
        
    elif type_ == '2c': # Q2 = R4 * D2 * R1
        y = resamp(x,4)
        
        # p0=resamp(y[:,:,::2,:],1)
        
        # inv(R4) * [1; 0] = [1; 1]
        # p1 = resamp(y([2:end, 1], 2:2:end), 1)
        p1 = torch.cat([y[...,1:,1::2], y[...,0:1,1::2]], dim=2)
        # p1 = resamp(p1,1)
        # p0, p1 = new_resamp([y[:,:,::2,:], p1], 1)
        p0, p1 = new_resamp([y[...,::2], p1], 1)
        
    else:
        raise ValueError('Invalid argument type')

    return p0, p1

def ppdec_layer(x, type_):
    # TODO
    if type_ == 1:      # P1 = R1 * Q1 = D1 * R3
        # p0=resamp(x[:,::2,:,:],3)

        # R1 * [0; 1] = [1; 1]
        #p1=resamp(np.roll(x[1::2,:],-1,axis=1),3)
        p1 = torch.cat([x[...,1::2,1:], x[...,1::2,0:1]], dim=3)
        # p1=resamp(p1, 3)
        # p0, p1 = new_resamp([x[:,::2,:,:], p1], 3)
        p0, p1 = new_resamp([x[...,::2,:], p1], 3)
        
    elif type_ == 2:    # P2 = R2 * Q2 = D1 * R4
        # p0=resamp(x[:,::2,:,:],4)
        
        # R2 * [1; 0] = [1; 0]
        # p1=resamp(x[:,1::2,:,:],4)
        # p0, p1 = new_resamp([x[:,::2,:,:], x[:,1::2,:,:]], 4)
        p0, p1 = new_resamp([x[...,::2,:], x[...,1::2,:]], 4)
        
    elif type_ == 3:    # P3 = R3 * Q2 = D2 * R1
        # p0=resamp(x[:,:,::2,:],1)
        
        # R3 * [1; 0] = [1; 1]
        #p1=resamp(np.roll(x[:,1::2],-1,axis=0),1)
        # p1 = torch.cat([x[:,1:,1::2,:], x[:,0:1,1::2,:]], dim=1)
        p1 = torch.cat([x[...,1:,1::2], x[...,0:1,1::2]], dim=2)
        # p1=resamp(p1, 1)
        # p0, p1 = new_resamp([x[:,:,::2,:], p1], 1)
        p0, p1 = new_resamp([x[...,::2], p1], 1)
        
    elif type_ == 4:    # P4 = R4 * Q1 = D2 * R2
        # p0=resamp(x[:,:,::2,:],2)
        
        # R4 * [0; 1] = [0; 1]
        # p1=resamp(x[:,:,1::2,:],2)
        # p0, p1 = new_resamp([x[:,:,::2,:], x[:,:,1::2,:]], 2)
        p0, p1 = new_resamp([x[...,::2], x[...,1::2]], 2)
        
    else:
        raise ValueError('Invalid argument type')
    
    return p0, p1

def resamp(x, type_, shift=1,extmod='per'):
    if type_ in [1,2]:
        y=resampm(x,type_,shift)
        
    elif type_ in [3,4]:
        y = torch.transpose(x, 2, 3)
        y = resampm(y, type_-2, shift)
        y = torch.transpose(y, 2, 3)
        
    else:
        raise ValueError('The second input (type) must be one of {1, 2, 3, 4}')

    return y

total = 0
def resampm(x, type_, shift=1):
    tic = time()
    N,c,m,n=x.shape

    x = torch.reshape(x, [-1, c, m*n])

    z=np.zeros((m,n), dtype=np.int64)
    for j in range(n):
        if type_ == 1:
            k= (shift * j) % m
            
        else:
            k= (-shift * j) % m
            
        if k < 0:
            k=k + m
        t1 = np.arange(k, m)
        t2 = np.arange(k)
        z[:,j] = np.concatenate([t1, t2]) * n + j
        
    z = z.reshape(-1)
    z = torch.from_numpy(z)
    z = z.to(x.device)

    # y = tf.gather(x, z.astype(int), axis=1)
    # y = tf.reshape(y, [-1, m, n, c])
    z = z.reshape((1,1,-1))
    y = torch.gather(x, 2, z.expand(N,c,-1))
    y = torch.reshape(y, [-1, c, m, n])
    
    toc = time()
    global total
    total += (toc-tic)
    # print('This resamp takes:', toc-tic, 'sec. Current time cost on resamp:', total)
    return y

def pdfbrec_layer(xlo, xhi, pfilt=None, dfilt=None):
    
    xhi = dfbrec_layer(xhi)

    x = lprec_layer(xlo, xhi)
    return x

def new_fbrec_layer(y0,y1,f_,type1,type2,extmod='per'):

    sample = len(y0)
    y0 = torch.cat(y0, axis=1)
    y1 = torch.cat(y1, axis=1)
    ch = y0.shape[1] // sample

    p1 = -1 / (2**0.5) * (y1 + sefilter2_layer(y0,f_,f_,extmod))
    
    p0 = (2**0.5) * y0 + sefilter2_layer(p1,f_,f_,extmod,[1,1])
    
    # Polyphase reconstruction
    if type1 == 'q':
        # Quincunx polyphase reconstruction
        x = qprec_layer(p0,p1,type2)
        
    elif type1 == 'p':
        # Parallelogram polyphase reconstruction
        x = pprec_layer(p0,p1,type2)
        
    else:
        raise ValueError('Invalid argument type1')
    
    return [x[:,i*ch:(i+1)*ch] for i in range(sample)]

def dfbrec_layer(y):
    f = dfb_filter(y[0].device)

    if type(y) is not list:
        dir = y.shape[1] // 3
        y = [y[..., sp*3:(sp+1)*3] for sp in range(dir)]

    n = np.log2(len(y))
    n = int(n)

    # Flip back the order of the second half channels
    y[2 ** (n - 1):]=y[-1:2 ** (n - 1)-1:-1]

    # Undo backsampling
    def rebacksamp(y=None):
        # Number of decomposition tree levels
        n=np.log2(len(y))
        assert not (n != round(n) or n < 1), 'Input must be a cell vector of dyadic length'
        n=int(n)
        if n == 1:
            # One level, the reconstruction filterbank shoud be Q1r
            # Redo the first resampling (Q1r = R2 * D1 * R3)
            for k in range(2):
                y[k][...,::2]=resamp(y[k][...,::2],2)
                y[k][...,1::2]=resamp(y[k][...,1::2],2)
                y[k]=resamp(y[k],3)
        if n > 2:
            N=2 ** (n - 1)
            
            for k in range(1,2 ** (n - 2)+1):
                shift=2*k - (2 ** (n - 2) + 1)
                
                # The first half channels
                # y[2*k - 2]=resamp(y[2*k - 2],3,-shift)
                # y[2*k - 1]=resamp(y[2*k - 1],3,-shift)
                y[2*k - 2], y[2*k - 1] = new_resamp([y[2*k - 2], y[2*k - 1]],3,-shift)
                
                # The second half channels
                # y[2*k - 2 + N]=resamp(y[2*k - 2 + N],1,-shift)
                # y[2*k - 1 + N]=resamp(y[2*k - 1 + N],1,-shift)
                y[2*k - 2 + N], y[2*k - 1 + N] = new_resamp([y[2*k - 2 + N], y[2*k - 1 + N]],1,-shift)
        
        return y
    y=rebacksamp(y)
    
    if n == 1:
        # Simplest case, one level
        x=fbrec_layer(y[0],y[1],f,'q','1r','qper_col')

    elif n >= 2:
        for l in range(n,2,-1):
            y_old=y.copy()
            y=[None] * (2 ** (l - 1))
            
            # The first half channels use R1 and R2
            # for k in range(1,2 ** (l - 2)+1):
            #     i=(k - 1) % 2 + 1
            #     y[k-1]=fbrec_layer(y_old[2*k-1],y_old[2*k-2],f,'p',i,'per')
            odd = np.arange(1, 2 ** (l - 2)+1, 2)
            even = np.arange(2, 2 ** (l - 2)+1, 2)
            tmp = new_fbrec_layer([y_old[2*k-1] for k in odd],[y_old[2*k-2] for k in odd],f,'p',1,'per')
            for idx, k in enumerate(odd):
                y[k-1] = tmp[idx]

            tmp = new_fbrec_layer([y_old[2*k-1] for k in even],[y_old[2*k-2] for k in even],f,'p',2,'per')
            for idx, k in enumerate(even):
                y[k-1] = tmp[idx]

            # The second half channels use R3 and R4
            # for k in range(2 ** (l - 2) + 1,2 ** (l - 1)+1):
            #     i=(k - 1) % 2 + 3
            #     y[k-1]=fbrec_layer(y_old[2*k-1],y_old[2*k-2],f,'p',i,'per')
            odd += 2 ** (l - 2)
            even += 2 ** (l - 2)
            tmp = new_fbrec_layer([y_old[2*k-1] for k in odd],[y_old[2*k-2] for k in odd],f,'p',3,'per')
            for idx, k in enumerate(odd):
                y[k-1] = tmp[idx]

            tmp = new_fbrec_layer([y_old[2*k-1] for k in even],[y_old[2*k-2] for k in even],f,'p',4,'per')
            for idx, k in enumerate(even):
                y[k-1] = tmp[idx]

        # x0 = fbrec_layer(y[1],y[0],f,'q','2c','per')
        # x1 = fbrec_layer(y[3],y[2],f,'q','2c','per')
        x0, x1 = new_fbrec_layer([y[1], y[3]],[y[0], y[2]],f,'q','2c','per')

        # First level
        x = fbrec_layer(x0,x1,f,'q','1r','qper_col')
    
    return x

def fbrec_layer(y0,y1,f_,type1,type2,extmod='per'):

    p1 = -1 / (2**0.5) * (y1 + sefilter2_layer(y0,f_,f_,extmod))
    
    p0 = (2**0.5) * y0 + sefilter2_layer(p1,f_,f_,extmod,[1,1])
    
    # Polyphase reconstruction
    if type1 == 'q':
        # Quincunx polyphase reconstruction
        x = qprec_layer(p0,p1,type2)
        
    elif type1 == 'p':
        # Parallelogram polyphase reconstruction
        x = pprec_layer(p0,p1,type2)
        
    else:
        raise ValueError('Invalid argument type1')
    
    return x
    
def slice_2c(y):
    idx = []
    n = y.shape[3]//2
    for i in range(n):
        idx.extend([i, i+n])
    # y = tf.gather(y, idx, axis=2)
    idx = torch.tensor(idx, dtype=torch.long, device=y.device)
    N,C,H,W = y.shape
    idx = torch.reshape(idx, (1,1,1,-1))
    y = torch.gather(y, 3, idx.expand(N,C,H,-1))
    return y

def slice_1r(y):
    idx = []
    m = y.shape[2]//2
    for i in range(m):
        idx.extend([i, i+m])
    # y = tf.gather(y, idx, axis=1)
    idx = torch.tensor(idx, dtype=torch.long, device=y.device)
    N,C,H,W = y.shape
    idx = torch.reshape(idx, (1,1,-1,1))
    y = torch.gather(y, 2, idx.expand(N,C,-1,W))
    return y

def qprec_layer(p0, p1, type_='1r'):
    m,n = p0.shape[1:3]
    
    if type_ == '1r':   # Q1 = R2 * D1 * R3
        # y1 = resamp(p0,4)
        # y2 = resamp(p1,4)
        y1, y2 = new_resamp([p0, p1], 4)
        y2 = torch.cat([y2[...,-1:], y2[...,:-1]], dim=3)
        y = torch.cat([y1, y2], dim=2)
        
        y = slice_1r(y)
        x = resamp(y,1)
        
    elif type_ == '1c': # Q1 = R3 * D2 * R2
        # TODO
        y=np.zeros((m,2*n))
        y[:,::2]=resamp(p0,1)
        y[:,1::2]=resamp(p1,1)
        
        x=resamp(y,4)

    elif type_ == '2r': # Q2 = R1 * D1 * R4
        # TODO
        y=np.zeros((2*m,n))
        y[::2,:]=resamp(p0,3)
        y[1::2,:]=resamp(p1,3)
        
        x=resamp(y,2)
        
    elif type_ == '2c': # Q2 = R4 * D2 * R1

        # y1 = resamp(p0,2)
        # y2 = resamp(p1,2)
        y1, y2 = new_resamp([p0, p1], 2)
        y2 = torch.cat([y2[:,:,-1:,:], y2[:,:,:-1,:]], dim=2)
        y = torch.cat([y1, y2], dim=3)

        y = slice_2c(y)
        x = resamp(y,3)
        
    else:
        raise ValueError('Invalid argument type')
    
    return x

def pprec_layer(p0, p1, type_):

    if type_ == 1:      # P1 = R1 * Q1 = D1 * R3
        '''x=np.zeros((2*m,n))
        x[::2,:]=resamp(p0,4)
        x[1::2,:]=np.roll(resamp(p1,4),1,axis=1)  # double check'''
        
        # x1 = resamp(p0,4)
        # x2 = resamp(p1,4)
        x1, x2 = new_resamp([p0, p1], 4)
        x2 = torch.cat([x2[...,-1:], x2[...,:-1]], dim=3)
        x = torch.cat([x1, x2], dim=2)
        
        # x = Lambda(lambda y: slice_1r(y))(x)
        x = slice_1r(x)
        
    elif type_ == 2:    # P2 = R2 * Q2 = D1 * R4
        '''x=np.zeros((2*m,n))
        x[::2,:]=resamp(p0,3)
        x[1::2,:]=resamp(p1,3)'''
        
        # x1 = resamp(p0,3)
        # x2 = resamp(p1,3)
        x1, x2 = new_resamp([p0, p1], 3)
        x = torch.cat([x1, x2], dim=2)

        x = slice_1r(x)
        
    elif type_ == 3:    # P3 = R3 * Q2 = D2 * R1
        '''x=np.zeros((m,2*n))
        x[:,::2]=resamp(p0,2)
        x[:,1::2]=np.roll(resamp(p1,2),1,axis=0)  # double check'''

        # x1 = resamp(p0,2)
        # x2 = resamp(p1,2)
        x1, x2 = new_resamp([p0, p1], 2)
        x2 = torch.cat([x2[:,:,-1:,:], x2[:,:,:-1,:]], dim=2)
        x = torch.cat([x1, x2], dim=3)

        x = slice_2c(x)
        
    elif type_ == 4:    # P4 = R4 * Q1 = D2 * R2
        '''x=np.zeros((m,2*n))
        x[:,::2]=resamp(p0,1)
        x[:,1::2]=resamp(p1,1)'''

        # x1 = resamp(p0,1)
        # x2 = resamp(p1,1)
        x1, x2 = new_resamp([p0, p1], 1)
        x = torch.cat([x1, x2], dim=3)

        x = slice_2c(x)
        
    else:
        raise ValueError('Invalid argument type')
    
    return x

def pixelShuffle(x, h_factor=2, w_factor=2):
    N, C, H, W = x.size()
    x = x.view(N, h_factor, w_factor, C // (h_factor * w_factor), H, W)  # (N, bs, bs, C//bs^2, H, W)
    x = x.permute(0, 3, 4, 1, 5, 2).contiguous()  # (N, C//bs^2, H, bs, W, bs)
    x = x.view(N, C // (h_factor * w_factor), H * h_factor, W * w_factor)  # (N, C//bs^2, H * bs, W * bs)
    return x

def inv_pixelShuffle(x, h_factor=2, w_factor=2):
    N, C, H, W = x.size()
    x = x.view(N, C, H // h_factor, h_factor, W // w_factor, w_factor)  # (N, C, H//bs, bs, W//bs, bs)
    x = x.permute(0, 3, 5, 1, 2, 4).contiguous()  # (N, bs, bs, C, H//bs, W//bs)
    x = x.view(N, C * (h_factor * w_factor), H // h_factor, W // w_factor)  # (N, C*bs^2, H//bs, W//bs)
    return x


def show_coeff(show):
    plot = True
    if plot:
        import matplotlib.pyplot as plt
        # show = c.cpu().numpy()
        if isinstance(show, list):
            dir = len(show)
            sub = 2 if dir==2 else 4
            for idx in range(sub):
                x = show[idx]
                x = x.cpu().numpy()
                if show[0].ndim>=4:
                    x = x[0]
                if show[0].shape[1] in [1,3,4]:
                    x = np.transpose(x, [1,2,0]) * 10
                
                plt.subplot(2,sub>>1,idx+1)
                plt.imshow(x)
        
        else:
            x = show.cpu().numpy()
            if show.ndim>=4:
                x = x[0]
            C = 3 if show.shape[1] >= 3 else 1
            x = np.transpose(x[:C], [1,2,0]) * 2
            plt.imshow(x)
        plt.show()


def normalize_subband(subband):
    subband_np = subband.cpu().numpy()
    min_val = subband_np.min()
    max_val = subband_np.max()
    if max_val - min_val == 0:
        normalized = np.zeros_like(subband_np)
    else:
        normalized = (subband_np - min_val) / (max_val - min_val)
    normalized = (normalized * 255).astype(np.uint8)
    return normalized

def log_scale_subband(subband):
    subband_np = subband.cpu().numpy()
    subband_np = np.abs(subband_np) + 1e-8  # 避免 log(0)
    log_scaled = np.log(subband_np)
    min_val = log_scaled.min()
    max_val = log_scaled.max()
    log_scaled = (log_scaled - min_val) / (max_val - min_val)
    log_scaled = (log_scaled * 255).astype(np.uint8)
    return log_scaled

def contrast_stretching(subband_np):
    stretched = cv2.normalize(subband_np, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    return stretched.astype(np.uint8)

def apply_colormap(subband_np):
    if len(subband_np.shape) == 2 or subband_np.shape[2] == 1:
        colored = cv2.applyColorMap(subband_np, cv2.COLORMAP_JET)
    else:
        colored = subband_np
    return colored












def dice_coeff(inputs):
    # inputs: [B, T, H*W]
    pred = inputs[:, None, :, :]
    target = inputs[:, :, None, :]

    mask = pred.new_ones(pred.size(0), target.size(1), pred.size(2))
    mask[:, torch.arange(mask.size(1)), torch.arange(mask.size(2))] = 0

    a = torch.sum(pred * target, -1)
    b = torch.sum(pred * pred, -1) + 1e-12
    c = torch.sum(target * target, -1) + 1e-12
    d = (2 * a) / (b + c)
    d = (d * mask).sum() / mask.sum()
    return d

def cross_scan_fwd(x: torch.Tensor, in_channel_first=True, out_channel_first=True, scans=0):
    if in_channel_first:
        B, C, H, W = x.shape
        if scans == 0:
            y = x.new_empty((B, 4, C, H * W))
            y[:, 0, :, :] = x.flatten(2, 3)
            y[:, 1, :, :] = x.transpose(dim0=2, dim1=3).flatten(2, 3)
            y[:, 2:4, :, :] = torch.flip(y[:, 0:2, :, :], dims=[-1])
        elif scans == 1:
            y = x.view(B, 1, C, H * W).repeat(1, 4, 1, 1)
        elif scans == 2:
            y = x.view(B, 1, C, H * W).repeat(1, 2, 1, 1)
            y = torch.cat([y, y.flip(dims=[-1])], dim=1)
    else:
        B, H, W, C = x.shape
        if scans == 0:
            y = x.new_empty((B, H * W, 4, C))
            y[:, :, 0, :] = x.flatten(1, 2)
            y[:, :, 1, :] = x.transpose(dim0=1, dim1=2).flatten(1, 2)
            y[:, :, 2:4, :] = torch.flip(y[:, :, 0:2, :], dims=[1])
        elif scans == 1:
            y = x.view(B, H * W, 1, C).repeat(1, 1, 4, 1)
        elif scans == 2:
            y = x.view(B, H * W, 1, C).repeat(1, 1, 2, 1)
            y = torch.cat([y, y.flip(dims=[1])], dim=2)

    if in_channel_first and (not out_channel_first):
        y = y.permute(0, 3, 1, 2).contiguous()
    elif (not in_channel_first) and out_channel_first:
        y = y.permute(0, 2, 3, 1).contiguous()

    return y


def cross_merge_fwd(y: torch.Tensor, in_channel_first=True, out_channel_first=True, scans=0):
    if out_channel_first:
        B, K, D, H, W = y.shape
        y = y.view(B, K, D, -1)
        if scans == 0:
            y = y[:, 0:2] + y[:, 2:4].flip(dims=[-1]).view(B, 2, D, -1)
            y = y[:, 0] + y[:, 1].view(B, -1, W, H).transpose(dim0=2, dim1=3).contiguous().view(B, D, -1)
        elif scans == 1:
            y = y.sum(1)
        elif scans == 2:
            y = y[:, 0:2] + y[:, 2:4].flip(dims=[-1]).view(B, 2, D, -1)
            y = y.sum(1)
    else:
        B, H, W, K, D = y.shape
        y = y.view(B, -1, K, D)
        if scans == 0:
            y = y[:, :, 0:2] + y[:, :, 2:4].flip(dims=[1]).view(B, -1, 2, D)
            y = y[:, :, 0] + y[:, :, 1].view(B, W, H, -1).transpose(dim0=1, dim1=2).contiguous().view(B, -1, D)        
        elif scans == 1:
            y = y.sum(2)
        elif scans == 2:
            y = y[:, :, 0:2] + y[:, :, 2:4].flip(dims=[1]).view(B, -1, 2, D)
            y = y.sum(2)

    if in_channel_first and (not out_channel_first):
        y = y.permute(0, 2, 1).contiguous()
    elif (not in_channel_first) and out_channel_first:
        y = y.permute(0, 2, 1).contiguous()
    
    return y


def cross_scan1b1_fwd(x: torch.Tensor, in_channel_first=True, out_channel_first=True, scans=0):
    if in_channel_first:
        B, _, C, H, W = x.shape
        if scans == 0:
            y = torch.stack([
                x[:, 0].flatten(2, 3),
                x[:, 1].transpose(dim0=2, dim1=3).flatten(2, 3),
                torch.flip(x[:, 2].flatten(2, 3), dims=[-1]),
                torch.flip(x[:, 3].transpose(dim0=2, dim1=3).flatten(2, 3), dims=[-1]),
            ], dim=1)
        elif scans == 1:
            y = x.flatten(2, 3)
        elif scans == 2:
            y = torch.stack([
                x[:, 0].flatten(2, 3),
                x[:, 1].flatten(2, 3),
                torch.flip(x[:, 2].flatten(2, 3), dims=[-1]),
                torch.flip(x[:, 3].flatten(2, 3), dims=[-1]),
            ], dim=1)
    else:
        B, H, W, _, C = x.shape
        if scans == 0:
            y = torch.stack([
                x[:, :, :, 0].flatten(1, 2),
                x[:, :, :, 1].transpose(dim0=1, dim1=2).flatten(1, 2),
                torch.flip(x[:, :, :, 2].flatten(1, 2), dims=[1]),
                torch.flip(x[:, :, :, 3].transpose(dim0=1, dim1=2).flatten(1, 2), dims=[1]),
            ], dim=2)
        elif scans == 1:
            y = x.flatten(1, 2)
        elif scans == 2:
            y = torch.stack([
                x[:, 0].flatten(1, 2),
                x[:, 1].flatten(1, 2),
                torch.flip(x[:, 2].flatten(1, 2), dims=[-1]),
                torch.flip(x[:, 3].flatten(1, 2), dims=[-1]),
            ], dim=2)

    if in_channel_first and (not out_channel_first):
        y = y.permute(0, 3, 1, 2).contiguous()
    elif (not in_channel_first) and out_channel_first:
        y = y.permute(0, 2, 3, 1).contiguous()

    return y


def cross_merge1b1_fwd(y: torch.Tensor, in_channel_first=True, out_channel_first=True, scans=0):
    if out_channel_first:
        B, K, D, H, W = y.shape
        y = y.view(B, K, D, -1)
        if scans == 0:
            y = torch.stack([
                y[:, 0],
                y[:, 1].view(B, -1, W, H).transpose(dim0=2, dim1=3).flatten(2, 3),
                torch.flip(y[:, 2], dims=[-1]),
                torch.flip(y[:, 3].view(B, -1, W, H).transpose(dim0=2, dim1=3).flatten(2, 3), dims=[-1]),
            ], dim=1)
        elif scans == 1:
            y = y
        elif scans == 2:
            y = torch.stack([
                y[:, 0],
                y[:, 1],
                torch.flip(y[:, 2], dims=[-1]),
                torch.flip(y[:, 3], dims=[-1]),
            ], dim=1)
    else:
        B, H, W, K, D = y.shape
        y = y.view(B, -1, K, D)
        if scans == 0:
            y = torch.stack([
                y[:, :, 0],
                y[:, :, 1].view(B, W, H, -1).transpose(dim0=1, dim1=2).flatten(1, 2),
                torch.flip(y[:, :, 2], dims=[1]),
                torch.flip(y[:, :, 3].view(B, W, H, -1).transpose(dim0=1, dim1=2).flatten(1, 2), dims=[1]),
            ], dim=2)
        elif scans == 1:
            y = y
        elif scans == 2:
            y = torch.stack([
                y[:, :, 0],
                y[:, :, 1],
                torch.flip(y[:, :, 2], dims=[1]),
                torch.flip(y[:, :, 3], dims=[1]),
            ], dim=2)

    if out_channel_first and (not in_channel_first):
        y = y.permute(0, 3, 1, 2).contiguous()
    elif (not out_channel_first) and in_channel_first:
        y = y.permute(0, 2, 3, 1).contiguous()

    return y


class CrossScanF(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, in_channel_first=True, out_channel_first=True, one_by_one=False, scans=0):
        # x: (B, C, H, W) | (B, H, W, C) | (B, 4, C, H, W) | (B, H, W, 4, C)
        # y: (B, 4, C, H * W) | (B, H * W, 4, C)
        ctx.in_channel_first = in_channel_first
        ctx.out_channel_first = out_channel_first
        ctx.one_by_one = one_by_one
        ctx.scans = scans

        if one_by_one:
            B, K, C, H, W = x.shape
            if not in_channel_first:
                B, H, W, K, C = x.shape
        else:
            B, C, H, W = x.shape
            if not in_channel_first:
                B, H, W, C = x.shape
        ctx.shape = (B, C, H, W)

        _fn = cross_scan1b1_fwd if one_by_one else cross_scan_fwd
        # _fn = cross_scan_fwd
        y = _fn(x, in_channel_first, out_channel_first, scans)

        return y
    
    @staticmethod
    def backward(ctx, ys: torch.Tensor):
        # out: (b, k, d, l)
        in_channel_first = ctx.in_channel_first
        out_channel_first = ctx.out_channel_first
        one_by_one = ctx.one_by_one
        scans = ctx.scans
        B, C, H, W = ctx.shape

        ys = ys.view(B, -1, C, H, W) if out_channel_first else ys.view(B, H, W, -1, C)
        _fn = cross_merge1b1_fwd if one_by_one else cross_merge_fwd
        y = _fn(ys, in_channel_first, out_channel_first, scans)
        
        if one_by_one:
            y = y.view(B, 4, -1, H, W) if in_channel_first else y.view(B, H, W, 4, -1)
        else:
            y = y.view(B, -1, H, W) if in_channel_first else y.view(B, H, W, -1)

        return y, None, None, None, None
    

class MaskModule(nn.Module):
    def __init__(self, 
                 n_levs, 
                 channels,
                 num_tokens=6,
                 is_flat=False, 
                 scan_mode="2D-selective-scan",
                 weights=True,
                 ):
        super(MaskModule, self).__init__()
        self.n_levs = n_levs
        self.channels = channels
        self.is_flat = is_flat
        self.scan_mode = scan_mode
        self.weights = weights
        self.mask_token = nn.Parameter(torch.randn(2**n_levs+1, 4, num_tokens, channels).normal_(0, 0.01)) # 4, 4, T, C

        
        if weights:
            self.prob = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=3, padding=1),
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(channels, num_tokens, kernel_size=1)
            )
        
        self.ct = ContourDec(n_levs)
        self.DTS = DepthToSpace(2,2)
        self.ict = ContourRec()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels  # fan-out
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
                if m.bias is not None:
                    m.bias.data.zero_()
    
    def forward_mask(self, x, mask_token):
        B, C, H, W = x.size()
        mask_token = mask_token.expand(B, 4, -1, -1) # [B, 4, T, C]
        k = CrossScanF.apply(x, True, True, False, 0) # [B, 4, C, H*W]
        # k = x.view(B, -1, H*W) ##################### 把注意力增加维度
        attn =  mask_token @ k # [B, 4, T, H*W]
        attn = attn.sigmoid()
        # attn = attn.view(B, 4, -1, H*W) # [B, 4, T, H*W]
        return attn
    
    def forward_prob(self, x):
        x = CrossScanF.apply(x, True, True, False, 0) # [B, 4, C, H*W] # 是否要merge存疑
        list_x = []
        for i in range(4):
            slice_i = x[:, i, :, :] # [B, C, H*W]
            mask_probs = self.prob(slice_i)  # [B, T, 1]
            list_x.append(mask_probs)
        mask_probs = torch.stack(list_x, dim=1) # [B, 4, T, 1]
        mask_probs = mask_probs.softmax(2).unsqueeze(3) # [B, 4, T, 1, 1]
        return mask_probs
    
    def forward_train(self, x):
        xlo, xhi = self.ct(x) ### 多层轮廓波变换从这里修改，以dict形式返回
        # print(xlo.shape, xhi[2].shape)
        assert isinstance(xhi, list)
        xhi_l = [] 

        for i, xhi in enumerate(xhi):
            mask_xhi = self.forward_mask(xhi, self.mask_token[i]) # [B, 4, T, H*W]
            xhi = CrossScanF.apply(xhi, True, True, False, 0) # [B, 4, C, H*W]
            out_xhi = xhi.unsqueeze(2) * mask_xhi.unsqueeze(3) # [B, 4, T, C, H*W]
            # print(out_xhi.shape)
            # if self.weights:
            #     out_xhi = out_xhi * self.forward_prob(xhi) # [B, 4, T, C, H*W]
            # out_xhi = out_xhi.sum(2).sum(1) # [B, C, H*W] # 是否要merge存疑
            xhi_l.append(out_xhi)

        mask_xlo = self.forward_mask(xlo, self.mask_token[-1]) # [B, 4, T, H*W]
        xlo = CrossScanF.apply(xlo, True, True, False, 0)
        xlo_l = xlo.unsqueeze(2) * mask_xlo.unsqueeze(3)
        
        return xlo_l, xhi_l
    
    def forward(self, x):
        xlo_l, xhi_l = self.forward_train(x)
        return xlo_l, xhi_l
    


@MODELS.register_module()
class ContourletTransKD(nn.Module):
    """
    Args:
        n_levs(int): "Number of Contourlet Transform levels"
    """

    def __init__(self, channels, n_levs=4, num_tokens=6, weights=True, custom_mask=True, custom_mask_warmup=1000, pretrained=None, loss_weight=1., factor=8):
        super(ContourletTransKD, self).__init__()
        self.weights = weights
        self.custom_mask = custom_mask
        self.custom_mask_warmup = custom_mask_warmup
        self.loss_weight = loss_weight
        
        self.mask_modules = nn.ModuleList([
            MaskModule(channels=c//(factor*factor), n_levs=n_levs, num_tokens=num_tokens, weights=weights) for c in channels
        ])

        self.init_weights(pretrained)
        self.iter = 0
        self.ct = ContourDec(n_levs)
        self.ict = ContourRec()
        self.DepthToSpace = DepthToSpace(factor,factor)  # 之后改为参数

    def init_weights(self, pretrained=None):
        if pretrained is not None:
            ckpt = _load_checkpoint(pretrained, map_location=torch.device('cpu'))
            state_dict = {}
            for k, v in ckpt['state_dict'].items():
                if 'mask_modules' in k:
                    state_dict[k[k.find('mask_modules'):]] = v  # save every layer which include mask_modules prams in state_dict
            self.load_state_dict(state_dict, strict=True)

    def forward(self, y_s_list, y_t_list):
        if not isinstance(y_s_list, (list, tuple)) or not isinstance(y_t_list, (list, tuple)):
            y_s_list = (y_s_list,)
            y_t_list = (y_t_list,)
        assert len(y_s_list) == len(y_t_list) == len(self.mask_modules)

        losses = []

        for y_s, y_t, mask_module in zip(y_s_list, y_t_list, self.mask_modules):
            y_s = self.DepthToSpace(y_s)
            y_t = self.DepthToSpace(y_t)
            t_xlo_l, t_xhi_l = mask_module.forward_train(y_t) #####要考虑到数量部分，如果是16个，我们不能直接遍历。
            # if self.custom_mask and self.iter >= self.custom_mask_warmup:
            #     if self.iter == self.custom_mask_warmup:
            #         print("warmup finished")
            #     with torch.no_grad():
            #         s_xlo, s_xhi_l = mask_module.forward_train(y_s) ##### 两个mask没法相乘计算
            with torch.no_grad():
                s_xlo_l, s_xhi_l = mask_module.forward_train(y_s) ##### ignore warmup
                    

            t_xhi_l = torch.stack(t_xhi_l, dim=0)
            s_xhi_l = torch.stack(s_xhi_l, dim=0) # [N, B, 4, T, C, H*W]
            # print(t_xhi_l.shape, s_xhi_l.shape)

            loss_H = F.mse_loss(s_xhi_l, t_xhi_l, reduction='none') # [N, B, 4, T, C, H*W]
            loss_L = F.mse_loss(s_xlo_l, t_xlo_l, reduction='none') 
            if self.weights:
                loss_H = loss_H * mask_module.forward_prob(y_t).unsqueeze(0)
                loss_L = loss_L * mask_module.forward_prob(y_t).unsqueeze(0)
            # print(loss)
            loss = loss_H.mean() + loss_L.mean()
            losses.append(loss)

        loss_total = sum(losses)

            ### 低通子带还没有写 v1.0
            ### 同样的手法完成了低通子带的计算

        return loss_total * self.loss_weight

    

if __name__ == '__main__':
    x1 = torch.randn(4,256,400,400)
    x2 = torch.rand(4,256,400,400)
    model = ContourletTransKD([256])
    loss = model(x1, x2)
    print(loss)

    # loss2 = F.mse_loss(x1, x2, reduction='mean')  
    # print(loss2)




            

            



            








    
