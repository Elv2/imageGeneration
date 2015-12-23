--require 'nn'
--require 'inn'
--require 'cudnn'
require 'nngraph'
require 'loadcaffe'
require 'Sampler'

if debug_output then nngraph.setDebug(true) end
local backend_name = 'nn'

--fc_inputMapSize = {192,8,8} cifar10
--fc_inputMapSize = {192,32,32}
fc_size = fc_inputMapSize[1]*fc_inputMapSize[2]*fc_inputMapSize[3]

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

local function initWeight(model)
	for k,v in pairs(model:findModules(('%s.SpatialConvolution'):format(backend_name))) do
		v.weight:normal(0,0.05)
		v.bias:zero()
	end
	for k,v in pairs(model:findModules(('%s.SpatialConvolutionMM'):format(backend_name))) do
		v.weight:normal(0,0.05)
		v.bias:zero()
	end
end

function createModel()
	-- load caffemodel
	--prototxt = '/home/tt3/DATA/caoqingxing/caffe/models/cifar10_nin/train_val.prototxt'
	--caffemodel = '/home/tt3/DATA/caoqingxing/caffe/models/cifar10_nin/cifar10_nin.caffemodel'
	--prototxt = '/home/tt3/DATA/caoqingxing/caffe/models/bvlc_alexnet/train_val.prototxt'
	--caffemodel = '/home/tt3/DATA/caoqingxing/caffe/models/bvlc_alexnet/bvlc_alexnet.caffemodel'
	--testNet = loadcaffe.load(prototxt,caffemodel,'nn')
	--testNet.modules[24] = nn.View(10)
	--testNet:add(nn.View(10))

	-- Encoder
	local x = nn.Identity()() -- Input image

	local torchmodel = 'pretrain-model/nin_caffe/model.net' -- load torch training model
	local encoder = torch.load(torchmodel)
--	encoder.modules[15] = nn.SpatialAveragePooling(3,3,2,2):ceil()
	local classifier = nn.Sequential()
	for i = encoder:size()-3,encoder:size() do -- remove last 4 layer of NIN model, 
		classifier:add(encoder:get(i):clone())
	end
	classifier.modules[3] = inn.SpatialAveragePooling(fc_inputMapSize[2],fc_inputMapSize[3],1,1)--:ceil()
	for i = encoder:size()-3,encoder:size() do
		encoder:remove()  
	end
	local encoderFC = encoder(x)  --Output feature map for cifar10&NIN is 8*8*192

	-- Latent Z
	local latent_input = nn.ConcatTable():add(nn.Linear(fc_size,latent_size)):add(nn.Linear(fc_size,latent_size))
	local z_mean,z_log_square_var = latent_input(nn.View(fc_size)(encoderFC)):split(2)
	local z = nn.Sampler()({z_mean, z_log_square_var})

	-- Classification
	local classifierOutput = classifier(encoderFC)
	--local classifierSoftmax = nn.SoftMax()(classifierOutput)
	local classifierSoftmax = nn.Identity()()

	-- Decoder
	local decoder = nn.Sequential()
	decoder:add(nn.JoinTable(1,1))
	decoder:add(nn.Linear(latent_size+10,fc_size))
	decoder:add(nn.View(fc_inputMapSize[1],fc_inputMapSize[2],fc_inputMapSize[3]))
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
	local x_prediction, x_prediction_var = decoder({z,classifierSoftmax}):split(2)

	nngraph.annotateNodes()
	local model = nn.gModule({x,classifierSoftmax},{classifierOutput,z_mean,z_log_square_var,x_prediction,x_prediction_var})

	if Nonlinear == 'Tanh' then
		local threshold_nodes, container_nodes = model:findModules('nn.ReLU')
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
	if modelInit then initWeight(model) end
	
	model:cuda()
	model.name = 'NIN_debug'
	-- debug
	if debug_output then
		paths.mkdir('debug')
		local input = torch.rand(3,32,32)
		pcall(function() model:updateOutput(input) end)
		graph.dot(model.fg, 'Forward Graph', 'debug')
	end
	
	return model,decoder
end