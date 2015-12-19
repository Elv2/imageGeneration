--require 'mobdebug'.start()

require 'nn'
require 'inn'
--require 'cudnn'
require 'nngraph'
require 'optim'
require 'image'
require 'loadcaffe'
require 'cifar.torch/provider'
local c = require 'trepl.colorize'

-- setting
latent_size = 256
opt = lapp[[
   -s,--save                  (default "logs/TanhNoNorm")      subdirectory to save logs
   -b,--batchSize             (default 128)          batch size
   -r,--learningRate          (default 3e-4)        learning rate
   --learningRateDecay        (default 1e-7)      learning rate decay
   --weightDecay              (default 0.0005)      weightDecay
   -m,--momentum              (default 0.9)         momentum
   --epoch_step               (default 25)          epoch step
   --max_epoch                (default 300)           maximum number of iterations
]]

nngraph.setDebug(true)


local function inverse_layer(x)
  local z
  
  if string.find(x.__typename, 'SpatialConvolution') ~= nil then
	z = nn.SpatialFullConvolution(x.nOutputPlane, x.nInputPlane, x.kW, x.kH, x.dW, x.dH, x.padW, x.padH)
  elseif string.find(x.__typename, 'SpatialMaxPooling') ~= nil then
    local scale = torch.floor(32 / ((32  + 2*x.padW - x.kW) / x.dW + 1))
    z = nn.SpatialUpSamplingNearest(scale)
	-- z = nn.SpatialFullConvolution(96, 96, x.kW, x.kH, x.dW, x.dH, x.padW, x.padH) -- NIN nInputPlane hard coding 96, modify it
  elseif string.find(x.__typename, 'SpatialAveragePooling') ~= nil then
    local scale = torch.floor(32 / ((32 - x.kW) / x.dW + 1))
    z = nn.SpatialUpSamplingNearest(scale)
	-- z = nn.SpatialFullConvolution(192, 192, x.kW, x.kH, x.dW, x.dH)
  elseif string.find(x.__typename, 'ReLU') ~= nil then
    z = nn.ReLU()
  else
    --print('not support inverse ' .. x.__typename .. ' layer, skip.' )
	return nil
  end
  return z
end

-- load caffemodel
--prototxt = '/home/tt3/DATA/caoqingxing/caffe/models/cifar10_nin/train_val.prototxt'
--caffemodel = '/home/tt3/DATA/caoqingxing/caffe/models/cifar10_nin/cifar10_nin.caffemodel'
--prototxt = '/home/tt3/DATA/caoqingxing/caffe/models/bvlc_alexnet/train_val.prototxt'
--caffemodel = '/home/tt3/DATA/caoqingxing/caffe/models/bvlc_alexnet/bvlc_alexnet.caffemodel'
--testNet = loadcaffe.load(prototxt,caffemodel,'nn')
--testNet.modules[24] = nn.View(10)
--testNet:add(nn.View(10))

-- load torch training model
x = nn.Identity()()

torchmodel = 'pretrain-model/nin_caffe/model.net'
encoder = torch.load(torchmodel)
for i = 1,4 do
  encoder:remove()  -- remove last 4 layer of NIN model, output feature map is 8*8*192
end

--print(encoder)
latent = nn.Linear(192*8*8,latent_size)(nn.View(192*8*8)(encoder(x)))
decoder_input = nn.View(192,8,8)(nn.Linear(latent_size,192*8*8)(latent))

decoder = nn.Sequential()

for i = encoder:size(),1,-1 do
   local z = inverse_layer(encoder:get(i));
   if z ~= nil then decoder:add(z) end
end
--print(decoder)
x_prediction = decoder(decoder_input)
d = nn.CSubTable()({x, x_prediction})
d2 = nn.Power(2)(d)
loss_xh = nn.Sum(4)(d2)
loss_xwh = nn.Sum(3)(loss_xh)
loss_x = nn.Sum(2)(loss_xwh)

nngraph.annotateNodes()
model = nn.gModule({x},{x_prediction,loss_x})

-- threshold_nodes, container_nodes = model:findModules('nn.ReLU')
-- for i = 1, #threshold_nodes do
  -- -- Search the container for the current threshold node
  -- for j = 1, #(container_nodes[i].modules) do
    -- if container_nodes[i].modules[j] == threshold_nodes[i] then
      -- -- Replace with a new instance
      -- container_nodes[i].modules[j] = nn.Tanh()
    -- end
  -- end
-- end

model:cuda()
-- data
datafile = '/home/tt3/DATA/caoqingxing/text2image/cifar.torch/providerMeanStdNorm.t7'
provider = torch.load(datafile)
provider.trainData.data = provider.trainData.data:cuda()
provider.testData.data = provider.testData.data:cuda()

-- debug
-- model.name = 'NIN_debug'
-- local input = torch.rand(3,32,32)
-- pcall(function() net:updateOutput(input) end)
-- train
print('Will save at '..opt.save)
paths.mkdir(opt.save)
testLogger = optim.Logger(paths.concat(opt.save, 'test.log'))
testLogger:setNames{'% mean class accuracy (train set)', '% mean class accuracy (test set)'}
testLogger.showPlot = false

parameters,gradParameters = model:getParameters()

print(c.blue'==>' ..' configuring optimizer')
optimState = {
  learningRate = opt.learningRate,
  weightDecay = opt.weightDecay,
  momentum = opt.momentum,
  learningRateDecay = opt.learningRateDecay,
}

function train()
  model:training()
  epoch = epoch or 1
  training_loss = 0;

  -- drop learning rate every "epoch_step" epochs
  if epoch % opt.epoch_step == 0 then optimState.learningRate = optimState.learningRate/2 end
  
  print(c.blue '==>'.." online epoch # " .. epoch .. ' [batchSize = ' .. opt.batchSize .. ']')

  local indices = torch.randperm(provider.trainData.data:size(1)):long():split(opt.batchSize)
  -- remove last element so that all the batches have equal size
  indices[#indices] = nil
  --debug
  -- for i = 1,380 do
    -- indices[#indices] = nil
  -- end
  -------
  local tic = torch.tic()
  for t,v in ipairs(indices) do
    xlua.progress(t, #indices)

    local inputs = provider.trainData.data:index(1,v)
	local df_dx = torch.zeros(inputs:size(1), 3, 32, 32):cuda()
	local df_dl = torch.ones(inputs:size(1), 1):cuda()

    local feval = function(x)
      if x ~= parameters then parameters:copy(x) end
      gradParameters:zero()
	  
      local x_predictions, loss_xs = unpack(model:forward(inputs))
      model:backward(inputs, {df_dx, df_dl})
	  
	  gradParameters:clamp(-5, 5)
      return loss_xs,gradParameters
    end
    _, batch_loss = optim.sgd(feval, parameters, optimState)
	training_loss = training_loss + torch.sum(batch_loss[1]:double())
  end
  
  training_loss = training_loss / provider.trainData.data:size(1)
  print(('Train accuracy: '..c.cyan'%f'..' \t time: %.2f s'):format(
        training_loss, torch.toc(tic)))
		
  epoch = epoch + 1
end

-- test
function test()
  -- disable flips, dropouts and batch normalization
  model:evaluate()
  print(c.blue '==>'.." testing")
  local bs = 125
  test_loss = 0;
  for i=1,provider.testData.data:size(1),bs do
    local x_predictions, loss_xs = unpack(model:forward(provider.testData.data:narrow(1,i,bs)))
    test_loss = test_loss + torch.sum(loss_xs:double())
  end
  test_loss = test_loss / provider.testData.data:size(1)
  print('Test accuracy:', test_loss)
  
  if testLogger or epoch > 2 then
    paths.mkdir(opt.save)
    testLogger:add{training_loss, test_loss}
    testLogger:style{'-','-'}
    testLogger:plot()

    local base64im
    do
      os.execute(('convert -density 200 %s/test.log.eps %s/test.png'):format(opt.save,opt.save))
      os.execute(('openssl base64 -in %s/test.png -out %s/test.base64'):format(opt.save,opt.save))
      local f = io.open(opt.save..'/test.base64')
      if f then base64im = f:read'*all' end
    end

    local file = io.open(opt.save..'/report.html','w')
    file:write(([[
    <!DOCTYPE html>
    <html>
    <body>
    <title>%s - %s</title>
    <img src="data:image/png;base64,%s">
    <h4>optimState:</h4>
    <table>
    ]]):format(opt.save,epoch,base64im))
    for k,v in pairs(optimState) do
      if torch.type(v) == 'number' then
        file:write('<tr><td>'..k..'</td><td>'..v..'</td></tr>\n')
      end
    end
    file:write'</table><pre>\n'
    file:write(tostring(test_loss)..'\n')
    file:write(tostring(model)..'\n')
    file:write'</pre></body></html>'
    file:close()
	
	if 2 == epoch then
	  graph.dot(model.fg, 'Forward Graph', opt.save..'/ForwardGraph')
      graph.dot(model.bg, 'Backward Graph', opt.save..'/BackwardGraph')
	end
  end

  -- save model every 50 epochs
  if epoch % 5 == 0 then
    local filename = paths.concat(opt.save, 'model.net')
    print('==> saving model to '..filename)
    torch.save(filename, model)
  end

end

train()
test()

for i=2,opt.max_epoch do
  train()
  test()
end
-------------------------------------------------------------------------------------------------------------------
-- test
-- confusion = optim.ConfusionMatrix(10)
-- testNet:evaluate()

-- local bs = 125
-- for i=1,dataset.testData.data:size(1),bs do
-- xlua.progress(i, dataset.testData.data:size(1))
-- outputs = testNet:forward(dataset.testData.data:narrow(1,i,bs))
-- confusion:batchAdd(outputs, dataset.testData.labels:narrow(1,i,bs)) --outputs[{{},{},1,1}]
-- end

-- confusion:updateValids()
-- print('Test accuracy:', confusion.totalValid * 100)
-------------------------------------------------------------------------------------------------------------------

-- testNet = loadcaffe.load(prototxt,caffemodel,'nn')
-- -- input = nn.Identity()()
-- -- latent = encoder(input)

-- -- testNet = nn.gModule({input},{latent})
-- testNet:cuda()

-- -- data
-- -- trainset = torch.load('cifar10-train.t7')
-- testset = torch.load('cifar10-test.t7')
-- testset.data = testset.data:cuda()
-- -- train
-- -- test
-- confusion = optim.ConfusionMatrix(10)
-- testNet:evaluate()

-- correct = 0
-- for i=1,10000 do
    -- local groundtruth = testset.label[i]
    -- local prediction = testNet:forward(testset.data:narrow(1,i,1))
    -- local confidences, indices = torch.sort(prediction, true)  -- true means sort in descending order
    -- if groundtruth == indices[1] then
        -- correct = correct + 1
    -- end
-- end

-- print(correct, 100*correct/10000 .. ' % ')