require 'cudnn'


function convLayer(a)
   return cudnn.SpatialConvolution(a)
end

function maxPool(a)
   return cudnn.SpatialMaxPooling(a)
end

function cuLinear(size1,size2)
   return Linear(size1,size2)
end

function Drop(a)
   return nn.Dropout(a)
end

function lSMLayer()
   return cudnn.LogSoftMax()
end

function buildAlexNet()
   torch.setdefaulttensortype('torch.FloatTensor')

   local cnn = nn.Sequential()
   print(cnn)
   return cnn
end


------------------load images & classes ----
local train_dir = '/hdd2/datasets/imagenet/train256max/'
local test_dir = '/hdd2/datasets/imagenet/val/'

train_set = torch.load(train_dir)
test_set = torch.load(test_dir)

local classes = {'food','bird'}

cnn = buildAlexNet()

cutorch.setDevice(3)

cnn = cnn:cuda()

------------Loss Function -----

crit = nn.ClassNLLCriterion()
crit = crit:cuda()

-------train NN-----

trainIt = nn.StochasticGradient(cnn, crit)
trainIt.learningRate = 0.001
trainIt.maxIteration = 5
train_set.data = train_set.data:cuda()

trainIt.train(train_set)

---------prediction ---------

for i=1,#test_set do
   pred = cnn:forward(test_set.data[i])
   print(pred:exp())
end

