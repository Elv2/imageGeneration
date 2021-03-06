--require 'mobdebug'.start()

require 'nn'
require 'inn'
require 'optim'
require 'image'

-- setting
modelInit = true
debug_output = false
latent_size = 256
dataNorm  = '01'
Nonlinear = 'ReLU'
opt = lapp[[
   -s,--save_prefix           (default "logs/clothCVAE")      subdirectory to save logs
   -b,--batchSize             (default 8)          batch size
   -r,--learningRate          (default 3e-6)        learning rate
   --learningRateDecay        (default 1e-7)      learning rate decay
   --weightDecay              (default 0.0005)      weightDecay
   -m,--momentum              (default 0.9)         momentum
   --epoch_step               (default 1)          epoch step
   --max_epoch                (default 300)           maximum number of iterations
]]
opt.save = opt.save_prefix .. Nonlinear .. dataNorm .. 'Norm/'

require 'model/NIN'
require 'clothDataset'
require 'KLDCriterion'
require 'GaussianCriterion'
local c = require 'trepl.colorize'

torch.manualSeed(1)

model, decoder = createModel()

-- data
datafile = 'cache/cloth_dataset.t7'
if paths.filep(datafile) then
	provider = torch.load(datafile)
	provider:initLoad()
else
	provider = DataSet()
	torch.save(datafile,provider)
end

-- train
print('Will save at '..opt.save)
paths.mkdir(opt.save)
testLogger = optim.Logger(paths.concat(opt.save, 'test.log'))
testLogger:setNames{'MCE (train set)','CE (train set)','KL (train set)', 'MCE (test set)','CE (test set)','KL (test set)'}
testLogger.showPlot = false

gen_folder = opt.save .. 'test/generate/'
gt_folder = opt.save .. 'test/gt/'
paths.mkdir(gen_folder)
paths.mkdir(gt_folder)

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

function generate()
  -- disable flips, dropouts and batch normalization
	decoder:evaluate()
	print(c.blue '==>'.." generating")
	local bs = 1
	test_loss = 0
	testSize = 100
	for i=1,testSize,1 do
		xlua.progress(i,testSize)
		z = torch.randn(1, latent_size):cuda()
		softmax = torch.zeros(1,10):cuda()
		softmax[{ {}, {1,9} }]:copy(provider.trainData.labels:index(1,i))
		x_prediction,x_prediction_var = unpack(decoder:forward({z,softmax}))
		im = x_prediction:clone()
		if dataNorm == 'MeanStd' then
			for channel=1,3 do -- over each image channel
				im[{ {channel}, {}, {}  }]:mul(provider.trainData.std[channel]) -- std scaling
				im[{ {channel}, {}, {}  }]:add(provider.trainData.mean[channel]) -- mean subtraction
			end
			image.save(gen_folder .. tostring(i) .. '.jpg', im:div(255))
		else
			image.save(gen_folder .. tostring(i) .. '.jpg', im)
		end
	end
end

function train()
  model:training()
  epoch = epoch or 1
  training_loss = torch.zeros(3);

  -- drop learning rate every "epoch_step" epochs
  if epoch % opt.epoch_step == 0 then optimState.learningRate = optimState.learningRate/2 end
  
  print(c.blue '==>'.." online epoch # " .. epoch .. ' [batchSize = ' .. opt.batchSize .. ']')

  local targets = torch.CudaTensor(opt.batchSize)
  local inputTargets = torch.CudaTensor(opt.batchSize,10)
  local indices = torch.randperm(provider.trainData.size()):long():split(opt.batchSize)
  -- remove last element so that all the batches have equal size
  indices[#indices] = nil

  local tic = torch.tic()
  for t,v in ipairs(indices) do
    xlua.progress(t, #indices)

	local inputs = torch.Tensor(opt.batchSize,3,128,128)
	for i = 1,(#v)[1] do
	  -- load new sample
	  inputs[i] = image.scale(provider.trainData.data[i], 128, 128)
	end
	inputs = inputs:cuda()
	--debug
	--targets:copy(provider.trainData.labels:index(1,v))
	targets:fill(1)
	inputTargets[{ {},{1,9} }]:copy(provider.trainData.labels:index(1,v))
	inputTargets[{ {},{10} }] = 0
    -----------------------------------------------------
    local feval = function(x)
      if x ~= parameters then parameters:copy(x) end
      gradParameters:zero()
	  
      local classifierOutput,z_mean,z_log_square_var,x_prediction,x_prediction_var = unpack(model:forward(inputs,inputTargets))
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
	
	if t % 1000 == 0 then
		generate()
		print(('Reconstruct negative log-likelihood:\t'..c.cyan'%f'..
		 '\nLatent negative log-likelihood:\t\t'..c.cyan'%f'..
		 '\nLatent mean: '..c.cyan'%f'..
		 '\ntime: %.2f s'):format(
		 training_loss[2]/t/opt.batchSize, training_loss[3]/t/opt.batchSize, z_means:mean(), torch.toc(tic)))
	end
	-- debug: one iteration for code checking
    --break
  end
  
  confusion:updateValids()
  train_acc = confusion.totalValid * 100
  
  training_loss:div(provider.trainData.size())
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
    local filename = paths.concat(opt.save, 'model.netWeights')
    print('==> saving model to '..filename)
    saveModelWeights(filename, model)
	
	filename = paths.concat(opt.save, 'decoder.netWeights')
	saveModelWeights(filename, decoder)
  end

end

for i=1,opt.max_epoch do
  train()
  --test()
  generate()
end
