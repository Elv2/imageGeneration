require 'nn'
require 'image'
require 'xlua'

local DataSet = torch.class 'DataSet'

function DataSet:__init(full)
  local source = '/home/caoqingxing/crawler/cloth_test/female_images/female_formatted_attributes_part_random_index.txt'
  local impath = {}
  local attri = {}
  local sampleCount = 0
  local attriCount = -1
  
  -- load image path text file
  io.input(source)
  for line in io.lines() do
	xlua.progress(sampleCount,253983)
	sampleCount = sampleCount+1
	attriCount = -1
	attri[sampleCount] = {}
	for word in string.gmatch(line,'[^\t]+') do
		attriCount = attriCount+1
		if attriCount == 0 then impath[sampleCount] = word
		else attri[sampleCount][attriCount] = tonumber(word) end
	end
  end
  -- load dataset
  self.trainData = {
	 data = {},
     impath_rootFolder = '/home/caoqingxing/crawler/cloth_test/female_images/',
     labels = torch.Tensor(attri),
	 imPath = impath,
     size = function() return sampleCount end
  }

  self.indices = torch.randperm(sampleCount)
  local trainData = self.trainData
  local indices = self.indices
  -- Mean Std Normalise
  if paths.filep('cache/meanfile.t7') then
	  meanStdv = torch.load('cache/meanfile.t7')
	  trainData.mean = meanStdv[1]
	  trainData.std = meanStdv[2]
	  print('Loaded Mean and Std')
  else
	  print('Normalising')
	  mean = {} -- store the mean, to normalize the test set in the future
	  stdv = {} -- store the standard-deviation for the future
	  data = torch.Tensor(10000, 3, 256, 256)
	  
	  for i = 1,10000 do
		 data[i] = image.load(trainData.impath_rootFolder..trainData.imPath[indices[i]])
	  end
	  for i=1,3 do -- over each image channel
		  mean[i] = data[{ {}, {i}, {}, {}  }]:mean() -- mean estimation
		  stdv[i] = data[{ {}, {i}, {}, {}  }]:std() -- std estimation
	  end

	  trainData.mean = mean
	  trainData.std = stdv
	  
	  paths.mkdir('cache')
	  torch.save('cache/meanfile.t7',{mean,stdv})
  end	
  collectgarbage()
end

function DataSet:initLoad()
  local trainData = self.trainData
  local indices = self.indices
  if dataNorm == '01' then
  setmetatable(trainData.data, {__index = function(self, index)
                                       im = image.load(trainData.impath_rootFolder..trainData.imPath[index])
									   --if torch.max(im) <= 1.5 then im:div(255) end
                                       return im
                                    end})
  else
  setmetatable(trainData.data, {__index = function(self, index)
                                       im = image.load(trainData.impath_rootFolder..trainData.imPath[index])
									   for channel=1,3 do
										im[{ {channel}, {}, {}  }]:add(-trainData.mean[channel])
										im[{ {channel}, {}, {}  }]:div(trainData.std[channel])
									   end
                                       return im
                                    end})
  end
end

function DataSet:shuffle()
	self.indices = torch.randperm(self.trainData.size())
end