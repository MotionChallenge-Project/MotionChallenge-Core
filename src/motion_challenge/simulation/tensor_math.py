"""Tensor quaternion math. Every public quaternion is XYZW."""
import torch

def normalize(q):
    return q/torch.linalg.vector_norm(q,dim=-1,keepdim=True).clamp_min(1e-15)

def conjugate(q):
    return torch.cat((-q[...,:3],q[...,3:]),dim=-1)

def multiply(a,b):
    av,bv=a[...,:3],b[...,:3]
    return torch.cat((a[...,3:]*bv+b[...,3:]*av+torch.linalg.cross(av,bv),
                      a[...,3:]*b[...,3:]-(av*bv).sum(-1,keepdim=True)),dim=-1)

def rotate(q,v):
    q=normalize(q)
    v,qv=torch.broadcast_tensors(v,q[...,:3])
    t=2*torch.linalg.cross(qv,v)
    return v+q[...,3:]*t+torch.linalg.cross(qv,t)

def matrix(q):
    x,y,z,w=normalize(q).unbind(-1)
    return torch.stack((1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w),
                        2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w),
                        2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)),dim=-1).reshape(*q.shape[:-1],3,3)

def slerp(a,b,f):
    a,b=normalize(a),normalize(b)
    dot=(a*b).sum(-1,keepdim=True)
    b=torch.where(dot<0,-b,b);dot=dot.abs().clamp(0,1)
    angle=dot.acos();den=angle.sin().clamp_min(1e-15)
    f=f[...,None]
    return torch.where(dot>.9995,normalize(a+f*(b-a)),
                       (((1-f)*angle).sin()*a+(f*angle).sin()*b)/den)

def quaternion_velocity(lo,hi,dt):
    q=multiply(conjugate(lo),hi);norm=torch.linalg.vector_norm(q[...,:3],dim=-1)
    angle=2*torch.atan2(norm,q[...,3]);angle=torch.where(angle>torch.pi,angle-2*torch.pi,angle)
    return q[...,:3]*(angle/norm.clamp_min(1e-15)/dt)[...,None]
