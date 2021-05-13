import torch
import numpy as np 
import torch.nn as nn
import torch.nn.functional as F 
import torch.optim as optim
import sys
import time

import copy
from lipschitz_lr import lipschitz_lr_calc,adjustLearningRate

globalPenulti = 10**18 #Arbitrary large integer. Not set to 0 because 1/Kz will tend to inf
globalUlti = 10**18

def ARELU(act):

	k = 0.6
	n = 1.2
	x = act-act
#	print(act)
	y = torch.where(act<=0,x,act)
	y = k*torch.pow(y,n)
	t = torch.where(act<0,x,y)
	return t

def leakyARELU(act):
	k = 0.6
	n = 1.2
	alpha = 0.01
	x = act-act
#	print(act)
	y = torch.where(act<=0,x,act)
	y = k*torch.pow(y,n)
	t = torch.where(act<0,alpha*y,y)
	return t

def norm(tensor):
    return torch.sqrt(torch.sum(torch.pow(tensor,2))).item()


def saveParams(model,epochNo):
	with open("lastEpoch.txt", "w+") as f:
		f.write(str(epochNo));
	fc1_weight = np.ravel(model.fc1.weight.data.detach().cpu().numpy())
	fc1_bias = np.ravel(model.fc1.bias.data.detach().cpu().numpy())
	fc2_weight = np.ravel(model.fc2.weight.data.detach().cpu().numpy())
	fc2_bias = np.ravel(model.fc2.bias.data.detach().cpu().numpy())
	
	params = np.concatenate((fc1_weight,fc1_bias,fc2_weight,fc2_bias))
	np.save("trainedData.npy",params)
class Network(nn.Module):
	def __init__(self,hidden_units,dropout_rate):
		super(Network,self).__init__()
		self.hidden_units=hidden_units
		self.fc1=nn.Linear(943,self.hidden_units)
		self.fc2=nn.Linear(self.hidden_units,4760)
		self.dropout=nn.Dropout(p=dropout_rate)

	def forward(self,x,actFunc):
		global globalPenulti,globalUlti
		
		x = copy.deepcopy(x)
		x = self.fc1(x)
		x = actFunc(x)
		
#		x = torch.relu(x)
		globalPenulti = x
#		x = torch.tanh(x)
		x = self.dropout(x)
		x = self.fc2(x)
		return x
def getbatch(bsize):
	seed = np.random.choice(X_train.size()[0], size = bsize, replace = False)
	xb = X_train[seed,:]
	yb = Y_train[seed,:]
	return xb,yb

def calculateMeanAndSTD(model,activationFunc,X_test,Y_test):
	vec = torch.abs(torch.sub(model.forward(X_test,activationFunc),Y_test)).detach().cpu().numpy()
	return np.mean(vec),np.std(vec)

def train(model,optimizer,steps,nRows,activationFunc,verbose=True,bsize=200,epochs=200): #increase batch size
	yTrainCost = norm(X_train)
	predictedLR=0
	for epoch in range(epochs):
		#call lipschitz_lr_calc followed by adjustLearningRate here
		print('curren LR',optimizer.param_groups[0]['lr'])
		for step in range(steps):
			x,y = getbatch(bsize)
			optimizer.zero_grad()
			outputs = model.forward(x,activationFunc)
			#loss = criterion(outputs,y)
			#loss = torch.sum(torch.sum(torch.pow(torch.sub(outputs,y),2),dim=0)/bsize)/n_outputs
			loss = torch.sum(torch.sum(torch.pow(torch.sub(outputs,y),2),dim=0)/bsize)#new loss according to paper large values of loss
			loss.backward()
			optimizer.step()
		#lossepoch = torch.sum(torch.sum(torch.pow(torch.sub(model.forward(X_train),Y_train),2),dim=0)/2921)/n_outputs
		lossepoch = torch.sum(torch.sum(torch.pow(torch.sub(model.forward(X_test,activationFunc),Y_test),2),dim=0))/(len(X_test))#new total loss
		MAE = torch.sum(torch.sum(torch.abs(torch.sub(model.forward(X_test,activationFunc),Y_test)),dim=0)/(len(X_test)))/n_outputs#this should be the MAE
		if verbose: print('epoch-{} training loss:{} MAE:{}'.format(epoch+1,lossepoch.item(),MAE.item()))
		if (epoch % 1)==0:
			with open('logfile.txt','a') as f:
				s= str(epoch) + '  ' +str(MAE.item())+' ' + str(optimizer.param_groups[0]['lr'])+'\n'
				f.write(s)
			
#			if (epoch%3) == 0:
#				saveParams(model,epoch)
	return MAE.item()
		

n_outputs = 4760
n_hidden = int(sys.argv[1])
dropout_rate = float(sys.argv[2])

Y_test = np.load('Ytest.npy')



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

X_test = torch.from_numpy(X_test).type(torch.FloatTensor).to(device)
Y_test = torch.from_numpy(Y_test).type(torch.FloatTensor).to(device)





funcFile = open('funcFileRedo_500.txt','w+')
f = open('logfileRedo_500.txt','w+')
f.write(str(n_hidden)+ '  '+str(dropout_rate)+'\n')
f.close()
finMAEVals = []
actFuncList = [ARELU,torch.tanh,torch.relu,torch.sigmoid]
#actFuncList = [leakyARELU]
funcStrList = ['ARELU','tanh','relu','sigmoid']



for func in actFuncList:
	finMAEVals = []
	for i in range(1,3):
		globalPenulti = 10**18
		Xstring = 'parsimonius_X_' + str(i) + '.npy'
		Ystring = 'parsimonius_Y_'+str(i) +'.npy'
		X_train = np.load(Xstring)
		Y_train = np.load(Ystring)
		X_train = torch.from_numpy(X_train).type(torch.FloatTensor).to(device)
		Y_train = torch.from_numpy(Y_train).type(torch.FloatTensor).to(device)
		model = Network(n_hidden,dropout_rate)
		model.to(device)
	#	model.__init__(n_hidden,dropout_rate)
		optimizer = optim.SGD(model.parameters(), lr=0.0001, momentum=0.5)
		MAE = train(model=model,optimizer = optimizer,activationFunc = func,nRows = len(X_train),steps=len(X_train)//200) # why 68
		finMAEVals.append(MAE)
		print(finMAEVals)
	funcName = funcStrList[actFuncList.index(func)]
#	mean,std = calculateMeanAndSTD(model,func,X_test,Y_test)
	funcFile.write(funcName + ' '+ str(np.mean(finMAEVals)) + '\n'+ str(np.std(finMAEVals)))


funcFile.close()
fc1_weight = np.ravel(model.fc1.weight.data.detach().cpu().numpy())
fc1_bias = np.ravel(model.fc1.bias.data.detach().cpu().numpy())
fc2_weight = np.ravel(model.fc2.weight.data.detach().cpu().numpy())
fc2_bias = np.ravel(model.fc2.bias.data.detach().cpu().numpy())

params = np.concatenate((fc1_weight,fc1_bias,fc2_weight,fc2_bias))


np.save("trainedData.npy",params)
print(params,params.shape)
