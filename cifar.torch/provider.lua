require 'nn'
require 'image'
require 'xlua'

local Provider = torch.class 'Provider'

function Provider:__init(full)
  local trsize = 50000
  local tesize = 10000

  -- download dataset
  if not paths.dirp('cifar-10-batches-t7') then
     local www = 'http://torch7.s3-website-us-east-1.amazonaws.com/data/cifar-10-torch.tar.gz'
     local tar = paths.basename(www)
     os.execute('wget ' .. www .. '; '.. 'tar xvf ' .. tar)
  end

  -- load dataset
  self.trainData = {
     data = torch.Tensor(50000, 3072),
     labels = torch.Tensor(50000),
     size = function() return trsize end
  }
  local trainData = self.trainData
  for i = 0,4 do
     local subset = torch.load('cifar-10-batches-t7/data_batch_' .. (i+1) .. '.t7', 'ascii')
     trainData.data[{ {i*10000+1, (i+1)*10000} }] = subset.data:t()
     trainData.labels[{ {i*10000+1, (i+1)*10000} }] = subset.labels
  end
  trainData.labels = trainData.labels + 1

  local subset = torch.load('cifar-10-batches-t7/test_batch.t7', 'ascii')
  self.testData = {
     data = subset.data:t():double(),
     labels = subset.labels[1]:double(),
     size = function() return tesize end
  }
  local testData = self.testData
  testData.labels = testData.labels + 1

  -- resize dataset (if using small version)
  trainData.data = trainData.data[{ {1,trsize} }]
  trainData.labels = trainData.labels[{ {1,trsize} }]

  testData.data = testData.data[{ {1,tesize} }]
  testData.labels = testData.labels[{ {1,tesize} }]

  -- reshape data
  trainData.data = trainData.data:reshape(trsize,3,32,32)
  testData.data = testData.data:reshape(tesize,3,32,32)
end

function Provider:normalize()
  ----------------------------------------------------------------------
  -- preprocess/normalize train/test sets
  --
  local trainData = self.trainData
  local testData = self.testData

  print '<trainer> preprocessing data (normalization)'
  collectgarbage()

  -- preprocess trainSet
  mean = {} -- store the mean, to normalize the test set in the future
  stdv = {} -- store the standard-deviation for the future
  for i=1,3 do -- over each image channel
      mean[i] = trainData.data[{ {}, {i}, {}, {}  }]:mean() -- mean estimation
      trainData.data[{ {}, {i}, {}, {}  }]:add(-mean[i]) -- mean subtraction
    
      stdv[i] = trainData.data[{ {}, {i}, {}, {}  }]:std() -- std estimation
      trainData.data[{ {}, {i}, {}, {}  }]:div(stdv[i]) -- std scaling
  end

  trainData.mean = mean
  trainData.std = stdv

  -- preprocess testSet
  for i=1,3 do -- over each image channel
      testData.data[{ {}, {i}, {}, {}  }]:add(-mean[i]) -- mean subtraction
	  testData.data[{ {}, {i}, {}, {}  }]:div(stdv[i]) -- std scaling
  end

end
