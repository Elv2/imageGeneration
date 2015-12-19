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
latent_size = 256
-- setting
opt = lapp[[
   --model                    (default "logs/VAEReLU01Norm/decoder.net")      Model used to generate image
   -s,--save                  (default "/test/")               Relative subdirectory to save images
]]

dataNorm = '01'
-- save directory
gen_folder = paths.dirname(opt.model) .. opt.save .. 'generate/'
gt_folder = paths.dirname(opt.model) .. opt.save .. 'gt/'
paths.mkdir(gen_folder)
paths.mkdir(gt_folder)
-- load torch training model
model = torch.load(opt.model)

-- data
datafile = '/home/tt3/DATA/caoqingxing/text2image/cifar.torch/provider'..dataNorm..'Norm.t7'
provider = torch.load(datafile)
provider.testData.data = provider.testData.data:cuda()

-- test
--confusion = optim.ConfusionMatrix(10)
Reconstruct_criterion = nn.GaussianCriterion():cuda()
-- disable flips, dropouts and batch normalization
model:evaluate()
print(c.blue '==>'.." testing")
local bs = 1
test_loss = 0
testSize = math.min(100,provider.testData.data:size(1))
for i=1,testSize,1 do
	xlua.progress(i,testSize)
	z = torch.randn(1, latent_size):cuda()
	softmax = torch.zeros(1,10):cuda()
	softmax[1][provider.testData.labels[i]] = 1
	local classifierOutput,z_mean,z_log_square_var,x_prediction,x_prediction_var = unpack(model:forward(provider.testData.data:narrow(1,i,bs)))

	local ReconstructLoss = Reconstruct_criterion:forward({x_prediction,x_prediction_var}, provider.testData.data:narrow(1,i,bs))

	test_loss = test_loss + ReconstructLoss
	--confusion:batchAdd(classifierOutput:select(1,1), provider.testData.labels:narrow(1,i,bs))

	im = x_prediction
	im_gt = provider.testData.data[i]:clone()
	if dataNorm == 'MeanStd' then
		for channel=1,3 do -- over each image channel
			im[{ {channel}, {}, {}  }]:mul(provider.trainData.std[channel]) -- std scaling
			im[{ {channel}, {}, {}  }]:add(provider.trainData.mean[channel]) -- mean subtraction

			im_gt[{ {channel}, {}, {}  }]:mul(provider.trainData.std[channel]) -- std scaling
			im_gt[{ {channel}, {}, {}  }]:add(provider.trainData.mean[channel]) -- mean subtraction
		end
		image.save(gen_folder .. tostring(i) .. '.jpg', im:div(255))
		image.save(gt_folder .. tostring(i) .. '.jpg', im_gt:div(255))
	else
		image.save(gen_folder .. tostring(i) .. '.jpg', im)
		image.save(gt_folder .. tostring(i) .. '.jpg', im_gt)
	end
end
test_loss = test_loss/testSize
--confusion:updateValids()
print('Test accuracy:', test_loss)

