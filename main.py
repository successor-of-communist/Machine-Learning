#!/usr/bin/env python
# coding=utf-8

import random
import numpy as np
from mpi4py import MPI
import math
import time

def getscore(up,vp):
    ret=np.dot(up,vp)
    if ret<1:
        return 1
    if ret>5:
        return 5
    return ret

def validate(st,ed,data,up,vp):
    rmse=0.0
    for i in range(st,ed):
        uid=data[i][3]
        iid=data[i][1]
        score=data[i][2]

        pscore=getscore(up.T[uid],vp.T[iid])
        rmse+=(score-pscore)*(score-pscore)

    return rmse

def dsadmm(comm,rank,size,P,start,testnum,pdata,pnum):

    k=40
    unum=943
    inum=1682
    rnum=100000
    up=np.random.random(size=(k+1,unum/P+1))
    vp=np.random.random(size=(k+1,inum+1))
    up/=math.sqrt(k)
    vp/=math.sqrt(k)
    v_ave=vp.copy()
    v_ave.dtype=vp.dtype

    sita=np.zeros((k+1,inum+1),dtype=np.float)
    alpha=0.002
    beta=0.7
    tao=0.1
    rou=0.05
    lambda1=0.05
    lambda2=0.05
    pdata=np.empty([rnum+1,4],dtype=int)
    #fi=open('p'+str(rank),'r')
    #for line in fi:
    #    arr=line.split()
    #    pdata[pnum][0]=int(arr[0].strip())
    #    pdata[pnum][1]=int(arr[1].strip())
    #    pdata[pnum][2]=int(arr[2].strip())
    #    pdata[pnum][3]=int(arr[3].strip())
    #    pnum+=1
    #fi.close()
    trainpnum=int(pnum*0.9)
    for step in range(1):
        for l in range(trainpnum):
            #userid=testdata[l][0]
            iid=pdata[l][1]
            score=pdata[l][2]
            uid=pdata[l][3]
            sigema=getscore(up.T[uid],vp.T[iid])
            #sigema-=up.T[prow]*vp.T[itemid]
            #for i in range(k):
            #    sigema+=up[i][prow]*vp[i][itemid]
            #if(sigema>5):
            #    sigema=5
            #if sigema<1:
            #    sigema=1
            sigema=score-sigema
            upi=[0 for i in range(k)]
            vpj=[0 for i in range(k)]
            for i in range(k):
                upi[i]=up[i][uid]+tao*(sigema*vp[i][iid]-lambda1*up[i][uid])
                vpj[i]=tao/(1+rou*tao)*((1-lambda2*tao)/tao*vp[i][iid]+sigema*up[i][uid]+rou*v_ave[i][iid]-sita[i][iid])
            for i in range(k):
                up[i][uid]=upi[i]
                vp[i][iid]=vpj[i]
        v_ave=comm.reduce(vp,root=0,op=MPI.SUM)
        if rank==0:
            v_ave/=P
        v_ave=comm.bcast(v_ave,root=0)
        curRmse=validate(trainpnum,pnum,pdata,up,v_ave)
        curRmse=comm.reduce(curRmse,root=0,op=MPI.SUM)
        sita=sita+rou*(vp-v_ave)
        if tao>alpha:
            tao=tao*beta
        if rank==0:
            curRmse=math.sqrt(curRmse/testnum)
            t=time.clock()-start
            print("test RMSE in step %d: %f   %f" %(step,curRmse,t))

if __name__=='__main__':
    start=time.clock()
    comm=MPI.COMM_WORLD
    rank=comm.Get_rank()
    size=comm.Get_size()
    testnum=0
    #if rank==0:
        #testnum=splitdata('./u.data',P,unum)
    splitdata=np.empty((P,rnum+1,4),int)
    upnum=unum/P+1;
    data=[0 for i in range(unum+1)]
    datarank=[0 for i in range(unum+1)]
    newrow=[0 for i in range(P)]
    rpnum=[0 for i in range(P)]
    for i in range(unum):
        rand=random.randint(0,P-1)
        while newrow[rand]>=upnum:
            rand=random.randint(0,P-1)
        else:
            data[i]=rand
            datarank[i]=newrow[rand]
            newrow[rand]+=1
    fi=open(datafile,'r')
    for line in fi:
        arr=line.split()
        uid=int(arr[0].strip())
        iid=int(arr[0].strip())
        rate=int(arr[0].strip())
        tmp=data[uid]
        splitdata[tmp][rpnum[tmp]]=[uid,iid,rate,datarank[uid]]
        rpnum[tmp]+=1
        #afi=open('p'+str(data[tmp]),'a')
        #afi.write(line.strip()+'\t'+str(datarank[tmp])+'\n')
        #afi.close()
    fi.close()
    testnum=0
    for i in range(P):
        testnum+=rpnum[i]-int(rpnum[i]*0.9)
    splitdata=comm.bcast(splitdata,root=0)
    dsadmm(comm,rank,size,P,start,testnum,splitdata[rank],rpnum[rank])
    end=time.clock()
    if rank==0:
        print(end-start)
