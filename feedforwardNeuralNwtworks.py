from pybrain.datasets               import ClassificationDataSet
from pybrain.utilities              import percentError
from pybrain.tools.shortcuts        import buildNetwork
from pybrain.supervised.trainers    import BackpropTrainer
from pybrain.structure.modules      import SoftmaxLayer
#from pylab import ion, ioff, figure, draw, contourf, clf, show, hold, plot
from pylab import * 
from scipy import diag, arange, meshgrid, where
from numpy.random import multivariate_normal

means = [(-1,0),(2,4),(3,1)]
cov = [diag([1,1]), diag([0.5, 1.2]), diag([1.5, 0.7])]
alldata = ClassificationDataSet(2, 1, nb_classes=3)
for n in range(400):
    for klass in range(3):
        input = multivariate_normal(means[klass],cov[klass])
        alldata.addSample(input, [klass])
tstdata_temp, trndata_temp = alldata.splitWithProportion(0.25)
tstdata = ClassificationDataSet(2, 1, nb_classes=3)
for n in range(0, tstdata_temp.getLength()):
    tstdata.addSample( tstdata_temp.getSample(n)[0], tstdata_temp.getSample(n)[1])

trndata = ClassificationDataSet(2, 1, nb_classes=3)
for n in range(0, trndata_temp.getLength()):
    trndata.addSample( trndata_temp.getSample(n)[0], trndata_temp.getSample(n)[1])

trndata._convertToOneOfMany()
tstdata._convertToOneOfMany()

print("Numer of traing patterns: ", len(trndata))
print( "Input and output dimensions:", trndata.indim, trndata.outdim)
print( "First sample (input, target,  class)")
print( trndata['input'][0], trndata['target'][0], trndata['class'][0])
