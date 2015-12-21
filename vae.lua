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
require 'KLDCriterion'
require 'GaussianCriterion'
require 'Sampler'


-- setting
local debug_output = false
local latent_size = 256
local dataNorm  = '01'
local Nonlinear = 'Tanh'
opt = lapp[[
   -s,--save_prefix           (default "logs/VAE")      subdirectory to save logs
   -b,--batchSize             (default 128)          batch size
   -r,--learningRate          (default 3e-4)        learning rate
   --learningRateDecay        (default 1e-7)      learning rate decay
   --weightDecay              (default 0.0005)      weightDecay
   -m,--momentum              (default 0.9)         momentum
   --epoch_step               (default 20)          epoch step
   --max_epoch                (default 300)           maximum number of iterations
]]
opt.save = opt.save_prefix .. Nonlinear .. dataNorm 'Norm'
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

if debug_output then nngraph.setDebug(true) end

torch.manualSeed(1)

-- load caffemodel
--prototxt = '/home/tt3/DATA/caoqingxing/caffe/models/cifar10_nin/train_val.prototxt'
--caffemodel = '/home/tt3/DATA/caoqingxing/caffe/models/cifar10_nin/cifar10_nin.caffemodel'
--prototxt = '/home/tt3/DATA/caoqingxing/caffe/models/bvlc_alexnet/train_val.prototxt'
--caffemodel = '/home/tt3/DATA/caoqingxing/caffe/models/bvlc_alexnet/bvlc_alexnet.caffemodel'
--testNet = loadcaffe.load(prototxt,caffemodel,'nn')
--testNet.modules[24] = nn.View(10)
--testNet:add(nn.View(10))

-- Encoder
x = nn.Identity()() -- Input image

torchmodel = 'pretrain-model/nin_caffe/model.net' -- load torch training model
encoder = torch.load(torchmodel)
classifier = nn.Sequential()
for i = encoder:size()-3,encoder:size() do -- remove last 4 layer of NIN model, 
	classifier:add(encoder:get(i):clone())
end
for i = encoder:size()-3,encoder:size() do
	encoder:remove()  
end
encoderFC = encoder(x)  --Output feature map for cifar10&NIN is 8*8*192

-- Latent Z
latent_input = nn.ConcatTable():add(nn.Linear(192*8*8,latent_size)):add(nn.Linear(192*8*8,latent_size))
z_mean,z_log_square_var = latent_input(nn.View(192*8*8)(encoderFC)):split(2)
z = nn.Sampler()({z_mean, z_log_square_var})

-- Classification
classifierOutput = classifier(encoderFC)
classifierSoftmax = nn.SoftMax()(classifierOutput)

-- Decoder
decoder = nn.Sequential()
decoder:add(nn.JoinTable(1,1))
decoder:add(nn.Linear(latent_size+10,192*8*8))
decoder:add(nn.View(192,8,8))
for i = encoder:size(),1,-1 do
	local z = inverse_layer(encoder:get(i));
	if z ~= nil then 
		if i == 1 then
			decoder:add(nn.ConcatTable():add(z):add(inverse_layer(encoder:get(i))))
		else
			decoder:add(z)
		end
	end
end
x_prediction, x_prediction_var = decoder({z,classifierSoftmax}):split(2)

nngraph.annotateNodes()
model = nn.gModule({x},{classifierOutput,z_mean,z_log_square_var,x_prediction,x_prediction_var})

if Nonlinear == 'Tanh' then
	threshold_nodes, container_nodes = model:findModules('nn.ReLU')
	for i = 1, #threshold_nodes do
		-- Search the container for the current threshold node
		for j = 1, #(container_nodes[i].modules) do
			if container_nodes[i].modules[j] == threshold_nodes[i] then
				-- Replace with a new instance
				container_nodes[i].modules[j] = nn.Tanh()
			end
		end
	end
end

--------

model:cuda()
model.name = 'NIN_debug'
-- debug
if debug_output then
	paths.mkdir('debug')
	local input = torch.rand(3,32,32)
	pcall(function() model:updateOutput(input) end)
	graph.dot(model.fg, 'Forward Graph', 'debug')
end

-- data
datafile = 'cifar.torch/provider'..dataNorm..'Norm.t7'
provider = torch.load(datafile)
provider.trainData.data = provider.trainData.data:cuda()
provider.testData.data = provider.testData.data:cuda()

-- train
print('Will save at '..opt.save)
paths.mkdir(opt.save)
testLogger = optim.Logger(paths.concat(opt.save, 'test.log'))
testLogger:setNames{'MCE (train set)','CE (train set)','KL (train set)', 'MCE (test set)','CE (test set)','KL (test set)'}
testLogger.showPlot = false

parameters,gradParameters = model:getParameters()

Classification_criterion = nn.CrossEntropyCriterion():cuda()
Reconstruct_criterion = nn.GaussianCriterion():cuda()
KLD = nn.KLDCriterion():cuda()
confusion = optim.ConfusionMatrix(10)


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
  training_loss = torch.zeros(3);

  -- drop learning rate every "epoch_step" epochs
  if epoch % opt.epoch_step == 0 then optimState.learningRate = optimState.learningRate/2 end
  
  print(c.blue '==>'.." online epoch # " .. epoch .. ' [batchSize = ' .. opt.batchSize .. ']')

  local targets = torch.CudaTensor(opt.batchSize)
  local indices = torch.randperm(provider.trainData.data:size(1)):long():split(opt.batchSize)
  -- remove last element so that all the batches have equal size
  indices[#indices] = nil

  local tic = torch.tic()
  for t,v in ipairs(indices) do
    xlua.progress(t, #indices)

    local inputs = provider.trainData.data:index(1,v)
	targets:copy(provider.trainData.labels:index(1,v))
    -----------------------------------------------------
    local feval = function(x)
      if x ~= parameters then parameters:copy(x) end
      gradParameters:zero()
	  
      local classifierOutput,z_mean,z_log_square_var,x_prediction,x_prediction_var = unpack(model:forward(inputs))
	  ------------------------------------------
	  -- debug fix sigma = 1; x_prediction_var = log(sigma^2)
	  x_prediction_var:fill(0)
	  ------------------------------------------
      local ClassifierLoss = Classification_criterion:forward(classifierOutput, targets)
      local df_do = Classification_criterion:backward(classifierOutput, targets)
	  local ReconstructLoss = Reconstruct_criterion:forward({x_prediction,x_prediction_var}, inputs)
      local df_dl = Reconstruct_criterion:backward({x_prediction,x_prediction_var}, inputs)
	  local KLDistance = KLD:forward(z_mean,z_log_square_var)
      local df_dzmu, df_dzvar = unpack(KLD:backward(z_mean,z_log_square_var))
	  
	  if z_means == nil then z_means = z_mean:clone()
	  else z_means:add(z_mean) end
	  ------------------------------------------
	  -- debug: Do not learn classifier & reconstruction sigma
	  df_do:fill(0)
	  df_dl[2]:fill(0)
	  ------------------------------------------
      model:backward(inputs, {df_do, df_dzmu, df_dzvar, df_dl[1], df_dl[2]})
	  
	  confusion:batchAdd(classifierOutput, targets)
	  
	  gradParameters:clamp(-5, 5)
      return {ClassifierLoss,ReconstructLoss,KLDistance},gradParameters
    end
    _, batch_loss = optim.sgd(feval, parameters, optimState)
	training_loss:add(torch.Tensor(batch_loss[1]))
  end
  
  confusion:updateValids()
  train_acc = confusion.totalValid * 100
  
  training_loss:div(provider.trainData.data:size(1))
  print(('Classification accuracy:\t\t'..c.cyan'%.2f'..
         '\nReconstruct negative log-likelihood:\t'..c.cyan'%f'..
		 '\nLatent negative log-likelihood:\t\t'..c.cyan'%f'..
		 '\nLatent mean: '..c.cyan'%f'..
		 '\ntime: %.2f s'):format(
        train_acc, training_loss[2], training_loss[3], z_means:mean(), torch.toc(tic)))

  confusion:zero()		
  epoch = epoch + 1
end

-- test
function test()
  -- disable flips, dropouts and batch normalization
  model:evaluate()
  print(c.blue '==>'.." testing")
  local bs = 125
  test_loss = torch.zeros(2);
  for i=1,provider.testData.data:size(1),bs do
    local classifierOutput,z_mean,z_log_square_var,x_prediction,x_prediction_var = unpack(model:forward(provider.testData.data:narrow(1,i,bs)))

	local ReconstructLoss = Reconstruct_criterion:forward({x_prediction,x_prediction_var}, provider.testData.data:narrow(1,i,bs))
	local KLDistance = KLD:forward(z_mean,z_log_square_var)

    test_loss:add(torch.Tensor({ReconstructLoss,KLDistance}))
	confusion:batchAdd(classifierOutput, provider.testData.labels:narrow(1,i,bs))
  end
  test_loss:div(provider.testData.data:size(1))
  confusion:updateValids()
  print('Test accuracy:', confusion.totalValid * 100, test_loss[1], test_loss[2])
  
  if testLogger or epoch > 2 then
    paths.mkdir(opt.save)
    testLogger:add{train_acc, training_loss[2], training_loss[3], confusion.totalValid * 100, test_loss[1], test_loss[2]}
    testLogger:style{'-','-','-','-','-','-'}
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
	file:write(tostring(confusion.totalValid * 100)..'\n')
	file:write(tostring(test_loss[1])..'\n')
    file:write(tostring(test_loss[2])..'\n')
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
	
	filename = paths.concat(opt.save, 'decoder.net')
	torch.save(filename, decoder)
  end

end

for i=1,opt.max_epoch do
  train()
  test()
end
