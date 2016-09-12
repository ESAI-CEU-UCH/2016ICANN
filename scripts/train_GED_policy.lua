--[[

  Copyright (C) 2015  Francisco Zamora-Martinez
  
  This program is free software: you can redistribute it and/or modify it under
  the terms of the GNU General Public License as published by the Free Software
  Foundation, either version 3 of the License, or (at your option) any later
  version.

  This program is distributed in the hope that it will be useful, but WITHOUT
  ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
  FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
  details.

  You should have received a copy of the GNU General Public License along with
  this program. If not, see <http://www.gnu.org/licenses/>.
  
  This script should be executed using APRIL-ANN toolkit:
  https://github.com/pakozm/april-ann

]]

april_print_script_header(arg)

local data_module = table.remove(arg, 1):match("^[^/]+")

----------------------------------------------------------
-- Loading data by requiring the module of MNIST-utils directory
local mnist_data = require(data_module)
-- unpacking data table into local variables
local train_input_ds, train_output_ds,
val_input_ds, val_output_ds,
tst_input_ds, tst_output_ds = table.unpack(mnist_data)
----------------------------------------------------------

local salt             = tonumber(arg[1] or 0.2) -- mask noise probability
local rpenalty         = tonumber(arg[2] or 0.2)  -- reconstruction penalty
local max_grads_norm   = tonumber(arg[3] or 6.5) -- max gradient norm
local max_norm_penalty = tonumber(arg[4] or 100) -- max norm penalty
local code_size        = tonumber(arg[5] or 2048)
local num_layers       = tonumber(arg[6] or 3)
local rpdecay          = tonumber(arg[7] or 0.999)
local filters          = arg[8] -- a prefix

local show_input_gradients = false
local rms_decay = 0.95

local ISIZE = train_input_ds:patternSize()
-- bunch_size controls the mini-batch for weight updates
local bunch_size       = 128
local wdecay           = 0.01
local rpenalty_decay   = rpdecay
-- replacement controls how many samples will be shown in one epoch
local replacement      = 256
----------------------------------------------------------

local rnd1 = random(1234)
local rnd2 = random(6543)
local rnd3 = random(8527)

-- noisy mask generator
local noise = salt>0 and stats.dist.bernoulli(1 - salt) or nil

print("# Generating MLP")

----------- AUTODIFF PART ------------

-- an MLP can be generated using autodiff package
local AD = autodiff
local op = AD.op
local S  = AD.scalar
local M  = AD.matrix
local T  = op.transpose
local logistic = AD.ann.logistic
local log_softmax = AD.ann.log_softmax
local CE_log_softmax  = AD.ann.cross_entropy_log_softmax
local CE_log_logistic = AD.ann.cross_entropy_log_logistic

local function declare_symbols_for_n_layers(n)
  local ws  = {}
  local bs  = {}
  local brs = {}
  local mhs = {}
  for i=1,n do
    ws[#ws+1] = AD.matrix('w%d'%{i})
    bs[#bs+1] = AD.matrix('b%d'%{i})
    brs[#brs+1] = AD.matrix('b%dr'%{i})
  end
  mhs[#mhs+1] = AD.matrix('mx')
  for i=1,n-1 do mhs[#mhs+1] = AD.matrix('mh%d'%{i}) end
  ws[#ws+1] = AD.matrix('w%d'%{n+1})
  bs[#bs+1] = AD.matrix('b%d'%{n+1})
  return ws,bs,brs,mhs
end

-- first, declare all needed variables
local rp,wd = AD.scalar('rp wd') -- rpenalty and wdecay in AD equations
local x  = AD.matrix('x')  -- clean input
local y  = AD.matrix('y')  -- target output
local ws,bs,brs,mhs = declare_symbols_for_n_layers(num_layers)

-- build Lua table with all parameters
local weights_table = {
  w1  = matrix(code_size, ISIZE), -- first layer filters
  b1  = matrix(code_size, 1),     -- first layer bias
  b1r = matrix(ISIZE, 1),         -- first layer reconstruction bias  
  --
  ['w%d'%{num_layers+1}] = matrix(10, code_size), -- last layer filters
  ['b%d'%{num_layers+1}] = matrix(10, 1),         -- last layer bias
}
for i=2,num_layers do
  weights_table['w%d'%{i}] = matrix(code_size, code_size)
  weights_table['b%d'%{i}] = matrix(code_size, 1)
  weights_table['b%dr'%{i}] = matrix(code_size, 1)
end

local weights_order = iterator(pairs(weights_table)):select(1):table()
table.sort(weights_order)

-- set broadcast of bias vectors
iterator(bs):apply(function(b) b:set_broadcast(false, true) end)
iterator(brs):apply(function(b) b:set_broadcast(false, true) end)

local function AE_subgoal(x, b0, w, b1, xp)
  xp = xp or x
  local hp = logistic( xp * T(w) + T(b1) )
  local o  = hp * w + T(b0)
  return CE_log_logistic( o, x )
end

-- compute ANN output
local N     = 1 -- op.dim(x, 1)
local hs    = {}
hs[#hs+1]   = x
for i=1,num_layers do hs[#hs+1] = logistic( hs[#hs] * T(ws[i]) + T(bs[i]) ) end
local o     = hs[#hs] * T(ws[#ws]) + T(bs[#bs])    -- forward propagation before softmax
local y_hat = log_softmax( o )                     -- log softmax output
local CE    = op.sum( CE_log_softmax( o, y ) ) / N -- cross entropy loss
local wreg  = op.sum(ws[#ws]^2) -- L2 regularization of last layer
local RE    = 0
for i=1,#mhs do
  local aux = op.cmul(hs[i], mhs[i]) 
  RE = RE + op.sum( AE_subgoal(hs[i], brs[i], ws[i], bs[i], aux) ) / N
end
local L = CE + rp * RE + wd * wreg -- global loss

-- compute differences plus loss
local dw_tbl = iterator{ ws, bs, brs }:iterate(ipairs):select(2):table()
local dL_dw = table.pack( L, AD.diff( L, dw_tbl ) )

local dh_tbl = iterator(hs):table()
local dL_dh  = table.pack( AD.diff( CE, dh_tbl ) )

-- compile functions
local forward    = AD.func( y_hat, { x }, weights_table ) -- returns ANN output
local backprop   = AD.func( dL_dw, { x, y, rp, wd, table.unpack(mhs) },
                            weights_table ) -- returns L + derivatives
local h_grad     = AD.func( dL_dh, { x, y }, weights_table ) -- hidden units grads

-- io.open("program.lua", "w"):write(backprop.program)
------------------------------------

local opt = ann.optimizer.adadelta()
opt:set_option("max_norm_penalty", max_norm_penalty)

local E_h_rms = {}
local train_dataset = function(in_ds, target_ds, weights_tbl)
  local mv = stats.running.mean_var()
  for input,target,indexes in trainable.dataset_multiple_iterator{
    datasets = { in_ds, target_ds },
    bunch_size = bunch_size,
    replacement = replacement,
    shuffle = rnd2,
  } do
    -- use the same mask in every optimizer iteration
    local masks = {}
    masks[#masks+1] = matrix(input:dim(1), input:dim(2))
    for i=1,num_layers-1 do
      masks[#masks+1] = matrix(input:dim(1), code_size)
    end
    if noise then
      iterator(masks):apply(function(m) noise:sample(rnd3, m:rewrap(m:size(),1)) end)
    else
      iterator(masks):apply(function(m) m:ones() end)
    end
    rpenalty = rpenalty * rpenalty_decay
    local loss = opt:execute(function(weights, it)
        assert(not it or it == 0,
               "Not allowed with line-search algorithms (like CG)")
        backprop:set_shared(weights)
        if show_input_gradients then
          local rms = function(m) return math.sqrt( m:dot(m)/m:size() ) end
          local dh = table.pack( h_grad( input, target, cache ) )
          local h_rms = iterator(dh):map(rms):table()
          for i=1,#h_rms do
            E_h_rms[i] = E_h_rms[i] or h_rms[i]
            E_h_rms[i] = rms_decay*E_h_rms[i] + (1-rms_decay)*h_rms[i]
          end
          print( "GRADS H_RMS", table.concat(h_rms," ") )
          print( "GRADS H_E_RMS", table.concat(E_h_rms," ") )
        end
        --
        local result = table.pack( backprop(input, target, rpenalty, wdecay,
                                            table.unpack(masks)) )
        local loss = table.remove(result, 1)
        local grads = iterator(ipairs(dw_tbl)):
        map(function(i,v) return v.name,result[i] end):table()
        --
        local norm2 = matrix.dict.norm2(grads)
        if norm2 > max_grads_norm and max_grads_norm > 0 then
          matrix.dict.scal(grads, max_grads_norm/norm2)
        end
        return loss, grads end,
      weights_tbl)
    for i=1,#indexes do mv:add(loss / #indexes) end
  end
  return mv:compute()
end

local validate_dataset = function(in_ds, target_ds, loss, logbase, weights_tbl)
  forward:set_shared(weights_tbl)
  loss:reset()
  for input,target,indexes in trainable.dataset_multiple_iterator{
    datasets = { in_ds, target_ds },
    bunch_size = math.max(512, bunch_size),
  } do
    local out = forward(input)
    if not logbase then out:exp() end
    loss:accum_loss(loss:compute_loss(out, target))
  end
  return loss:get_accum_loss()
end

local function neurons_max_norm2(weigths)
  return iterator(weigths):
    map(function(w)
        for _,sw in iterator(matrix.ext.iterate(w)) do
          coroutine.yield(sw:norm2())
        end
    end):max()
end

local uniform = function(w, r) w:uniformf(-r, r, rnd1) end

-- randomize the neural network weights (no biases) in the range
-- [ inf / sqrt(fanin + fanout), sup / sqrt(fanin + fanout) ]
for wname in iterator(weights_order) do
  local w = weights_table[wname]
  if wname:find("^w.*$") then
    uniform(w, math.sqrt(6 / (w:dim(1) + w:dim(2) + 1)))
  else
    w:zeros()
  end
end

local stopping_criterion = trainable.stopping_criteria.make_max_epochs_wo_imp_relative(2)
local pocket_alg = trainable.train_holdout_validation{
  min_epochs =   400,
  max_epochs = 10000,
  stopping_criterion = stopping_criterion,
}

print("# Epoch Train-CE Val-ER best_epoch best_val_error \t time/epoch norm2")
local cronometro = util.stopwatch()
cronometro:go()

-- train until pocket_alg:execute is false; trian_func uses the given stopping
-- criterion to decide when to return true or false
while pocket_alg:execute(
  function()
    local train_error = train_dataset(train_input_ds, train_output_ds,
                                      weights_table)
    local val_error   = validate_dataset(val_input_ds, val_output_ds,
                                         ann.loss.zero_one(), false,
                                         weights_table)
    return weights_table, train_error, val_error
end) do
  local epoch = pocket_alg:get_state_table().current_epoch
  local cpu,wall = cronometro:read()
  printf("%s \t cpu: %.2f wall: %.2f :: norm2  w= %8.4f  b= %8.4f\n",
         pocket_alg:get_state_string(),
  	 cpu/epoch, wall/epoch,
         neurons_max_norm2{ weights_table.w1, weights_table.w2,
                            weights_table.w3, weights_table.w4 },
         neurons_max_norm2{ weights_table.b1, weights_table.b2,
                            weights_table.b3, weights_table.b4 })
  io.stdout:flush()
end
cronometro:stop()
local cpu,wall = cronometro:read()
local epochs = pocket_alg:get_state_table().current_epoch
printf("# Wall total time: %.3f    per epoch: %.3f\n", wall, wall/epochs)
printf("# CPU  total time: %.3f    per epoch: %.3f\n", cpu, cpu/epochs)

-- take the best model and compute zero-one error (classification error)
local best = pocket_alg:get_state_table().best
local val_rel_error = validate_dataset(val_input_ds, val_output_ds,
                                       ann.loss.zero_one(), false, best)
local tst_rel_error = validate_dataset(tst_input_ds, tst_output_ds,
                                       ann.loss.zero_one(), false, best)
printf("# VAL  CLASS ERROR %.4f %%  %d\n",
       val_rel_error*100, val_rel_error*val_input_ds:numPatterns())
printf("# TEST CLASS ERROR %.4f %%  %d\n",
       tst_rel_error*100, tst_rel_error*tst_input_ds:numPatterns())

-- -- when an epoch is the best, show at screen the validation and test zero-one
-- -- errors (classification errors) which is (100 - accuracy)
-- if pocket_alg:is_best() then
--   local val_rel_error = pocket_alg:get_state_table().validation_error
--   local tst_rel_error = validate_dataset(tst_input_ds, tst_output_ds,
--                                          ann.loss.zero_one(), false,
--                                          weights_table)
--   printf("# VAL  CLASS ERROR %.4f %%  %d\n",
--          val_rel_error*100, val_rel_error*val_input_ds:numPatterns())
--   printf("# TEST CLASS ERROR %.4f %%  %d\n",
--          tst_rel_error*100, tst_rel_error*tst_input_ds:numPatterns())
-- end

if filters then
  -- save the filters
  local aux
  for _,wname in ipairs{"w1","w2","w3","w4", "w5"} do
    local w = best[wname]
    if not w then break end
    aux = (aux and w * aux) or w
    local img = ann.connections.input_filters_image(aux, {28, 28})
    ImageIO.write(img, string.format("%s.%s.png", filters, wname))
  end
end
