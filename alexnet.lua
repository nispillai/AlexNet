require 'nn'
require 'torch'
require 'cudnn'
require 'image'


function maxPool(a,b,c,d)
   return nn.SpatialMaxPooling(a,b,c,d)
end

function cuLinear(size1,size2)
   return nn.Linear(size1,size2)
end

function Drop(a)
   return nn.Dropout(a)
end

function lSMLayer()
   return nn.LogSoftMax()
end

function reLu() 
   return nn.ReLU(true)
end

function thresh(a,b)
   return nn.Threshold(a,b)   
end

function buildAlexNet(nClasses)
   torch.setdefaulttensortype('torch.FloatTensor')
   local cnn = nn.Sequential()
   local cnn2 = nn.Sequential()
   cnn:add(nn.SpatialConvolution(3,96,11,11,4,4,2,2))
   cnn:add(reLu())   
   cnn:add(maxPool(3,3,2,2))
   cnn:add(nn.SpatialConvolution(96,256,5,5,1,1,2,2))
   cnn:add(reLu())
   cnn:add(maxPool(3,3,2,2))
   cnn:add(nn.SpatialConvolution(256,384,3,3,1,1,1,1))
   cnn:add(reLu())
   cnn:add(nn.SpatialConvolution(384,384,3,3,1,1,1,1))
   cnn:add(reLu())
   cnn:add(nn.SpatialConvolution(384,256,3,3,1,1,1,1))
   cnn:add(reLu())   
   cnn:add(maxPool(3,3,2,2))
   cnn:cuda()
   
   local cnn1 = nn.Sequential()
   cnn1:add(nn.View(256*6*6))
   cnn1:add(Drop(0.5))
   cnn1:add(cuLinear(256*6*6,4096))
   cnn1:add(thresh(0,1e-6))   
   cnn1:add(Drop(0.5))
   cnn1:add(cuLinear(4096,4096))
   
   cnn1:add(thresh(0,1e-6))   
   cnn1:add(cuLinear(4096,nClasses)) 
   cnn1:add(lSMLayer())
   cnn1:cuda()
   
   local model = nn.Sequential():add(cnn):add(cnn1)
   return model
end

function getDirectories(dir,noDir)
   chk = 1
   dirNames = {}
   local p = io.popen('ls ' .. dir)
   for file in p:lines() do
      dirNames[chk] = file
      if chk == noDir then
         io.flush()
         return dirNames
      end
      chk = chk + 1
   end
   io.flush()
   return dirNames
end

function fileNumbers(dir,dirNames)
   fTot = 0
   for c = 1,#dirNames do
      file = dirNames[c]
      dir1  = dir .. "/" .. file
      local pq = io.popen('find "'..dir1..'" -type f')
      for fpq in pq:lines() do
         fTot = fTot + 1
      end
      io.flush()
   end
   return fTot
end
function dirLookup(dir,noDir)

   classes = {}
   ij = 1
   imTot = 1

   local dirNames = getDirectories(dir,noDir)
   local fTot = fileNumbers(dir,dirNames)
   local imagesAll = torch.Tensor(fTot,3,400,200)
   local labelsAll = torch.Tensor(fTot)

   local labelNo = 1
   for c = 1,#dirNames do
      file = dirNames[c]
      f1 = string.sub(file, 2)
      classes[ij] = f1
      ij = ij + 1

      dir1  = dir .. "/" .. file
      local pq = io.popen('find "'..dir1..'" -type f')
      for fpq in pq:lines() do
         ok = image.load(fpq)
         it = image.scale(ok,400,200)
         if it:size(1) == 3 then
            imagesAll[labelNo] = it
            labelsAll[labelNo] = f1
            labelNo = labelNo + 1
         end
      end
      io.flush()
    end
-- create train set:
   trainset = {
      data = torch.Tensor(labelNo, 3, 400, 200),
      label = torch.Tensor(labelNo),
      size = function() return labelNo end
   }
   setmetatable(trainset,
    {__index = function(t, i)
                    return {t.data[i], t.label[i]}
                end}
   );
  for ijk = 1,labelNo do
         trainset.data[ijk] = imagesAll[ijk]:clone()
         trainset.label[ijk] = labelsAll[ijk]
   end
   trainset.data = trainset.data:double()
   return classes,trainset
end




------------------load images & classes ----
local train_dir = '/hdd2/datasets/imagenet/train256max/'
local test_dir = '/hdd2/datasets/imagenet/val/'

local classes , trainset = dirLookup(train_dir,2)
---test_set = torch.load(test_dir)

print(classes)

cnn = buildAlexNet(1000)

cutorch.setDevice(3)

cnn = cnn:cuda()

print 'model:'
print(cnn)
os.exit()
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
