equire 'cudnn'


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
end


buildAlexNet()
