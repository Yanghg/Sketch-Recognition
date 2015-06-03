execfile("StrokeHMMbasic.py")
states = ['Sunny','Cloudy','Rainy']
features = ['Evidence']
contOrDisc={'Evidence':1}
numVals = {'Evidence':4}


hmm = HMM(states,features,contOrDisc,numVals)
hmm.priors = {'Sunny':0.63,'Cloudy':0.17,'Rainy':0.2}
hmm.emissions = {'Sunny':{'Evidence':[0.6,0.2,0.15,0.05]},'Cloudy':{'Evidence':[0.25,0.25,0.25,0.25]},'Rainy':{'Evidence':[0.05,0.10,0.35,0.50]}}
hmm.transitions = {'Sunny':{'Sunny':0.5,'Cloudy':0.375,'Rainy':0.125},'Cloudy':{'Sunny':0.25,'Cloudy':0.125,'Rainy':0.625},'Rainy':{'Sunny':0.25,'Cloudy':0.375,'Rainy':0.375}}
hmm.label([{'Evidence':0},{'Evidence':2},{'Evidence':3}])