"""GPU metric implementation, independently checked against NumPy/CPU."""
import torch
from motion_challenge.simulation.tensor_math import normalize,multiply,conjugate
NAMES=("mpkpe_world_m","mpkpe_xy_aligned_m","mpkpe_xy_yaw_aligned_m","joint_mae_rad","anchor_height_error_m","anchor_gravity_z_error")

def score(ap,aq,joints,rp,rq,ref_joints,root_row,body_rows,protocol):
    ap,rp=ap.double(),rp.double()
    aq,rq=normalize(aq.double()),normalize(rq.double())
    ar,rr=ap[:,root_row],rp[:,root_row]
    height=(ar[:,2]-rr[:,2]).abs()
    gravity=lambda q:2*(q[:,0]**2+q[:,1]**2)-1
    tilt=(gravity(aq)-gravity(rq)).abs()
    a,r=ap[:,body_rows],rp[:,body_rows]
    distance=lambda a,b:torch.linalg.vector_norm(a-b,dim=-1).mean(-1)
    world=distance(a,r)
    a=a.clone();r=r.clone();a[...,:2]-=ar[:,None,:2];r[...,:2]-=rr[:,None,:2]
    xy=distance(a,r)
    relative=normalize(multiply(rq,conjugate(aq)))
    x,y,z,w=relative.unbind(-1)
    cosine=2*(w*w-z*z);sine=4*z*w
    angle=torch.where(torch.hypot(cosine,sine)<=1e-12,0.,torch.atan2(sine,cosine))
    c,s=angle.cos()[:,None],angle.sin()[:,None]
    aligned=torch.stack((c*a[...,0]-s*a[...,1],s*a[...,0]+c*a[...,1],a[...,2]),dim=-1)
    values=torch.stack((world,xy,distance(aligned,r),(joints.double()-ref_joints.double()).abs().mean(-1),height,tilt),dim=-1)
    failure=(height>protocol["failure"]["anchor_height_m"])|(tilt>protocol["failure"]["gravity_z"])
    return values,failure
