require 'mobdebug'.start()

require 'nn'
require 'inn'
--require 'cudnn'
require 'nngraph'
require 'optim'
require 'image'
require 'loadcaffe'
require 'cifar.torch/provider'
local c = require 'trepl.colorize'
nngraph.setDebug(true)

-- setting
opt = lapp[[
   --model                    (default "logs/debug/model.net")      Model used to generate image
   -s,--save                  (default "/test/")               Relative subdirectory to save images
]]

-- save directory
gen_folder = paths.dirname(opt.model) .. opt.save .. 'generate/'
gt_folder = paths.dirname(opt.model) .. opt.save .. 'gt/'
paths.mkdir(gen_folder)
paths.mkdir(gt_folder)
-- load torch training model
model = torch.load(opt.model)

-- data
datafile = '/home/tt3/DATA/caoqingxing/text2image/cifar.torch/providerMeanStdNorm.t7'
provider = torch.load(datafile)
provider.testData.data = provider.testData.data:cuda()

-- test
-- disable flips, dropouts and batch normalization
model:evaluate()
print(c.blue '==>'.." testing")
local bs = 125
test_loss = 0;
for i=1,100 do
  if i > provider.testData.data:size(1) then break end
  xlua.progress(i,100)
  x_predictions, loss_xs = unpack(model:forward(provider.testData.data:narrow(1,i,1)))
  test_loss = test_loss + torch.sum(loss_xs:double())
  im = x_predictions
  im_gt = provider.testData.data[i]:clone()
  for channel=1,3 do -- over each image channel
	im[{ {channel}, {}, {}  }]:mul(provider.trainData.std[channel]) -- std scaling
    im[{ {channel}, {}, {}  }]:add(provider.trainData.mean[channel]) -- mean subtraction
	
	im_gt[{ {channel}, {}, {}  }]:mul(provider.trainData.std[channel]) -- std scaling
    im_gt[{ {channel}, {}, {}  }]:add(provider.trainData.mean[channel]) -- mean subtraction
  end
  image.save(gen_folder .. tostring(i) .. '.jpg', im:div(255))
  image.save(gt_folder .. tostring(i) .. '.jpg', im_gt:div(255))
end
test_loss = test_loss / provider.testData.data:size(1)
print('Test accuracy:', test_loss)

