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
   cnn:add(convLayer(3,96,11,11,4,4,2,2))
   cnn:add(maxPool(3,3,2,2))
   cnn:add(convLayer(96,256,5,5,1,1,2,2))
   cnn:add(maxPool(3,3,2,2))
   cnn:add(convLayer(256,384,3,3,1,1,1,1))
   cnn:add(convLayer(384,384,3,3,1,1,1,1))
   cnn:add(convLayer(384,256,3,3,1,1,1,1))
   cnn:add(maxPool(3,3,2,2))
   cnn:add(cuLinear(256*6*6,4086))
   cnn:add(Drop(0.5))
   cnn:add(cuLinear(4096,4096))
   cnn:add(Drop(0.5))a
   cnn:add(cuLinear(4096,1000)) 
   cnn:add(lSMLayer)

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
