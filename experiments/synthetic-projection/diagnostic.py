import json
import numpy as np
rng=np.random.default_rng(42); d=128; r=8; k=4
B=np.linalg.qr(rng.normal(size=(d,r)))[0]
def data(n):
 z=rng.normal(size=(n,r)); return z@B.T, (z[:,0]>=0).astype(int)
X,y=data(40); D,dy=data(100); T,ty=data(100)
mu=X.mean(0); X=X-mu; T=T-mu
_,s,vt=np.linalg.svd(X,full_matrices=False)
def rank(A):
 sv=np.linalg.svd(A,compute_uv=False); tol=sv[0]*max(A.shape)*np.finfo(A.dtype).eps
 return int((sv>tol).sum())
def score(Q):
 assert np.allclose(Q,Q.T,atol=1e-12) and np.allclose(Q@Q,Q,atol=1e-12)
 a=X@Q; b=T@Q; w=np.linalg.solve(a.T@a+0.1*np.eye(d),a.T@(2*y-1))
 pred=(b@w>=0).astype(int); assert len(pred)==len(ty)==100 and np.isfinite(w).all()
 return {'rank':rank(a),'correct':int((pred==ty).sum()),'total':len(ty),'accuracy':float((pred==ty).mean())}
I=np.eye(d); p=vt[:k].T; P=I-p@p.T
out={'synthetic_only':True,'train_n':len(y),'dev_n':len(dy),'test_n':len(ty),'unprojected':score(I),'pca_delete_4':score(P),'random_delete_4':[]}
assert out['unprojected']['rank']==8 and out['pca_delete_4']['rank']==4
for seed in range(10,20):
 q=np.linalg.qr(np.random.default_rng(seed).normal(size=(d,k)))[0]; v=score(I-q@q.T); assert v['rank']==8; out['random_delete_4'].append({'seed':seed,**v})
print(json.dumps(out,indent=2))
