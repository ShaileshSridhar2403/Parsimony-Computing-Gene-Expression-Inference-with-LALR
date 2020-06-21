import torch
import torch.nn as nn
import numpy as np 
from loadDataset import getSeqBatch


class Hook():
	def __init__(self, module, backward=False):
		if backward==False:
			self.hook = module.register_forward_hook(self.hook_fn)
		else:
			self.hook = module.register_backward_hook(self.hook_fn)
	def hook_fn(self, module, input, output):
		self.input = input
		self.output = output
	def close(self):
		self.hook.remove()

class Net(nn.Module):
	def __init__(self):
		super(Net,self).__init__()
		self.linear1 = nn.Linear(50,50)
		#replace with Arelu
		self.relu = nn.ReLU(inplace=True)
		self.linear2 = nn.Linear(50,3)

	def forward(self,x):
		x = self.linear1(x)
		x = self.relu(x)
		x = self.linear2(x)
		return x
	
def norm(tensor):
    return torch.sqrt(torch.sum(torch.pow(tensor,2))).item()

def lipschitz_lr_calc(model,bs,x_train,yTrainCost,globalPenulti,actFunc):  #pass len(x_train)
	Kz = 0
	Ka = 0
#	penultimate_activation_hook = Hook(list(model.children())[-3])
	for i in range(len(x_train)//bs):
		xtrain = getSeqBatch(x_train,bs,i)
		with torch.no_grad():
			final_layer_act = model.forward(xtrain,actFunc)
#			penultimate_act = penultimate_activation_hook.output.cpu().detach().numpy()
			final_layer_act = norm(final_layer_act)
#			penultimate_act = np.linalg.norm(penultimate_act)
			if type(globalPenulti)!=int:
				glob = norm(globalPenulti)
			else:
				glob = 10**18 #Arbitrary large integer. Not set to 0 because 1/Kz will tend to inf
			if final_layer_act>Ka:
				Ka = final_layer_act
#			if penultimate_act>Kz:
#				Kz = penultimate_act
			if glob>Kz:
				Kz = glob
	
	K = (Ka+yTrainCost)*Kz/len(x_train)
	print('Ka',Ka,'Kz',Kz,'glob',glob,'k',K,'yTrainCost',yTrainCost)
	lr = 0.5/K
	print('Ka',Ka,'Kz',Kz,'glob',glob,'lr',lr)
	return lr 




def adjustLearningRate(optimizer,newLR):
#	newLR = min(newLR,5*10**-3)
#	if newLR > 5*10**-4:
#		newLR = 5*10**-4
	if newLR < 1*10**-5:
		newLR = 1*10**-5
	for param_group in optimizer.param_groups:
		param_group['lr'] = newLR
		
def linearDecayLR(optimizer,decayFactor):
	for param_group in optimizer.param_groups:
		param_group['lr'] = max(decayFactor * param_group['lr'],10**(-5))
