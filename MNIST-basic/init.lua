--[[
  MNIST-REEMLPs APRIL-ANN tutorial
  Copyright (C) 2015  Francisco Zamora-Martinez

  This program is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  This program is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with this program.  If not, see <http://www.gnu.org/licenses/>.
]]

 -- data has to be in the same the path where the script is located
local datadir = "/home/experimentos/CORPORA/MNIST-basic"
local train_filename = "mnist_train.amat"
local test_filename  = "mnist_test.amat"

-- loads the training and test matrices
print("# Lodaing trainig data...")
local training_data = matrix.read(io.open(datadir.."/"..train_filename),
                                  { [matrix.options.tab]   = true,
                                    [matrix.options.ncols] = 785,
                                    [matrix.options.nrows] = 12000, })
print("# Lodaing test data...")
local test_data = matrix.read(io.open(datadir.."/"..test_filename),
                              { [matrix.options.tab]   = true,
                                [matrix.options.ncols] = 785,
                                [matrix.options.nrows] = 50000, })

assert(training_data:dim(2) == 785)
assert(test_data:dim(2) == 785)

local training_samples = training_data(':', '1:784'):clone()
local training_labels  = training_data(':', 785) + 1.0

local test_samples = test_data(':', '1:784'):clone()
local test_labels  = test_data(':', 785) + 1.0

-- the output is an indexed dataset over a identity which allows to produce a
-- local encoding
local identity = dataset.identity(10, 0.0, 1.0)

local function build_input_output_dataset(samples, labels)
  local input_ds = dataset.matrix(samples)
  local output_ds = dataset.indexed(dataset.matrix(labels), { identity })
  return input_ds, output_ds
end

-- generate training datasets
local train_input_data, train_output_data =
  build_input_output_dataset(training_samples, training_labels)

-- training partition (10000 samples)
local train_input  = dataset.slice(train_input_data,  1, 10000)
local train_output = dataset.slice(train_output_data, 1, 10000)

-- validation partition (2000 samples)
local validation_input  = dataset.slice(train_input_data,  10001, 12000)
local validation_output = dataset.slice(train_output_data, 10001, 12000)

-- generate test dataset
local test_input, test_output =
  build_input_output_dataset(test_samples, test_labels)

print("# Training size:   ", train_input:numPatterns())
print("# Validation size: ", validation_input:numPatterns())
print("# Test size:       ", test_input:numPatterns())

return { train_input, train_output,
         validation_input, validation_output,
         test_input, test_output }
