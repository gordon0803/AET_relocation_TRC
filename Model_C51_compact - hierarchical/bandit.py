import numpy as np
import time
import scipy
import numba as nb
import config

# @nb.jit
# def _return_upper_bound_batch(features,theta,alpha,AaI,n_action):
#     prob=[0 for i in range(len(features))]
#     count=0;
#     for feature in features:
#         tprob=[feature.dot(theta[i])+alpha*np.sqrt((feature.dot(AaI[i])).dot(feature)) for i in range(n_action)]
#         prob[count]=np.array(tprob)
#         count+=1
#     return np.array(prob)


class linucb_agent():
    def __init__(self,n_action,d):
        #number of actions: n_action
        #feature length: d
        self.alpha=1
        self.round=0;
        self.lmbda=.5; #penalty parameter for l2 regression
        self.d=int(d)
        self.n_action=n_action
        self.Aa=[] #collection of A for each arm
        self.ba=[] #collection of vectors to compute disjoint part d*1
        self.AaI=[] #inverse of A
        self.theta=[]
        self.arm_det=[]
        self.alpha_list=[]
        #initialize parameters
        for i in range(n_action):
            self.Aa.append(self.lmbda*np.identity(self.d))
            self.ba.append(np.zeros(self.d))
            self.AaI.append(np.identity(self.d))
            self.theta.append(np.zeros(self.d))
            self.arm_det.append(1)
            self.alpha_list.append(1)

    def update(self,features,actions,rewards):
        #update all observed arms
        #reset parameters
        #gamma=1; #decay parameter
       # for i in range(self.n_action):
        #    self.Da[i]=gamma*self.Da[i]
         #   self.ba[i]=gamma*self.ba[i]
        self.round+=len(rewards) #number of rounds the bandit has been played


        t1=time.time()
        features=np.vstack(features)
        act=np.vstack(actions.copy())
        #process action to make N_station being self
        for j in range(self.n_action):
            idx=act[:,j]==self.n_action
            act[idx,j]=j
        rewards=np.vstack(rewards)
        features_exp=np.tile(features,[self.n_action,1]) #expand the features map
        actions_exp=act.T.flatten()
        seq_id=np.tile([i for i in range(len(features))],self.n_action)
        rewards_exp=rewards[seq_id,actions_exp]

        for j in range(self.n_action):
            feature = features_exp[actions_exp == j][:,j*self.d:(j+1)*self.d]
            reward = rewards_exp[actions_exp == j]
            outer_feature = feature.T.dot(feature)
            outer_ba = feature.T .dot(reward)
            self.Aa[j] += outer_feature
            self.ba[j] += outer_ba
        # print('process 1:', time.time()-t1)

        # t1 = time.time()
        # for i in range(len(features)):
        #     f=features[i]
        #     for j in range(self.n_action): #each station actions
        #         action=actions[i][j]
        #         if action<self.n_action:
        #             self.Aa[action]+=f[:,None]*f[None,:]
        #             self.ba[action]+= rewards[i][action]*f
        # print('process 2:', time.time() - t1)
        #inverse doesn't have to be calculated for each feature
        for action in range(self.n_action):
            self.AaI[action]=scipy.linalg.inv(self.Aa[action]) #inverse
            self.theta[action]=np.dot(self.AaI[action],self.ba[action])
            self.arm_det[action]=np.log(np.sqrt(np.linalg.det(self.Aa[action]))*((np.linalg.det(self.lmbda*np.identity(self.d)))**-0.5)*10)
            self.alpha_list[action]=2*np.sqrt(2*self.arm_det[action])+np.sqrt(self.lmbda)


        # self.alpha=0.01


        #self.alpha=0.05*np.sqrt(self.d*np.log(1+self.round*10*1/0.1))+np.sqrt(0.1)
        # self.alpha=0.01
        #print(self.alpha,self.round)

        #done update

    def return_upper_bound(self,feature):
        # prob=[]
       # s=np.array(feature)
        #
        # for i in range(self.n_action):
        #     score=np.dot(s,self.theta[i])+self.alpha*np.sqrt(np.dot(np.dot(s,self.AaI[i]),s))
        #     prob.append(score)
        s=feature
        # prob=[np.dot(s,self.theta[i])+self.alpha*np.sqrt(np.dot(np.dot(s,self.AaI[i]),s)) for i in range(self.n_action)]

        prob=np.fromiter((np.dot(s[i*self.d:(i+1)*self.d],self.theta[i])+self.alpha_list[i]*np.sqrt(np.dot(np.dot(s[i*self.d:(i+1)*self.d],self.AaI[i]),s[i*self.d:(i+1)*self.d])) for i in range(self.n_action)), float)


        # prob=_return_upper_bound(s,self.theta,self.alpha,self.AaI,self.n_action)

        return np.array(prob)

    def return_upper_bound_batch(self,feature):
        feature=np.vstack(feature)
        # prob=np.array([np.einsum('ij,j->i',feature,self.theta[i])+self.alpha*np.sqrt(np.einsum('ij,jj,ij->i',feature,self.AaI[i],feature)) for i in range(self.n_action)]).T
        prob = np.array([np.einsum('ij,j->i', feature[:,i*self.d:(i+1)*self.d], self.theta[i]) + self.alpha_list[i] * np.sqrt(np.einsum('ij,jj,ij->i', feature[:,i*self.d:(i+1)*self.d], self.AaI[i], feature[:,i*self.d:(i+1)*self.d])) for i in range(self.n_action)]).T


        # prob=[self.return_upper_bound(feature[i]) for i in range(len(feature))]



        return np.array(prob)

    def measure_error(self,feature,score,threshold):
        feature=np.vstack(feature)
        score=np.vstack(score)
        mean=np.array([np.einsum('ij,j->i', feature[:,i*self.d:(i+1)*self.d], self.theta[i]) for i in range(self.n_action)]).T
        bound=np.array([self.alpha_list[i] * np.sqrt(np.einsum('ij,jj,ij->i', feature[:,i*self.d:(i+1)*self.d], self.AaI[i], feature[:,i*self.d:(i+1)*self.d])) for i in range(self.n_action)]).T
        #print(mean[1:3,:],bound[1:3,:],score[1:3,:])
        count=np.asarray(mean+bound+0*np.sqrt(2*0.1)*scipy.special.erfinv(-0.9)<threshold,float)*np.asarray(score>threshold,float)
        eliminate_total=score>threshold
        arm_err=[]
        for st in range(self.n_action):
            arm_err.append(count[:,st].sum()/eliminate_total[:,st].sum())
        err_total=count.sum()
        err=err_total/eliminate_total.sum()
        return err,arm_err


    def return_regret(self,feature,score,threshold,confidence):
        err,arm_err=self.measure_error(feature,score,threshold)
        if err>confidence+np.sqrt(np.log(10)/(2*len(score)*self.n_action)):
            switch=1
        else:
            switch=0
        return err,switch,arm_err







