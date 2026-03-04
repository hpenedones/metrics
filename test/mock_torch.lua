-- Minimal torch mock for testing roc.lua without the full Torch dependency.
-- Supports only the tensor operations used by roc.lua and roc tests.

local torch = {}

------------------------------------------------------------
-- Tensor class
------------------------------------------------------------
local Tensor = {}
Tensor.__index = Tensor

-- Create a Tensor from a flat table or with given dimensions
function torch.Tensor(...)
   local args = {...}
   local t = setmetatable({}, Tensor)
   if type(args[1]) == "table" then
      -- torch.Tensor({1,2,3}) -> 1D tensor
      t._data = {}
      for i, v in ipairs(args[1]) do t._data[i] = v end
      t._rows = #t._data
      t._cols = nil
   elseif #args == 1 then
      -- torch.Tensor(n) -> 1D tensor of zeros
      t._data = {}
      t._rows = args[1]
      t._cols = nil
      for i = 1, t._rows do t._data[i] = 0 end
   elseif #args == 2 then
      -- torch.Tensor(rows, cols) -> 2D tensor of zeros
      t._rows = args[1]
      t._cols = args[2]
      t._data = {}
      for i = 1, t._rows do
         t._data[i] = {}
         for j = 1, t._cols do t._data[i][j] = 0 end
      end
   end
   return t
end

function torch.DoubleTensor(tbl)
   return torch.Tensor(tbl)
end

function torch.IntTensor(tbl)
   return torch.Tensor(tbl)
end

-- Size helper object
local Size = {}
Size.__index = Size

function Size.new(dims)
   return setmetatable({ _dims = dims }, Size)
end

function Size:size()
   return #self._dims
end

Size.__index = function(self, k)
   if type(k) == "number" then return self._dims[k] end
   return Size[k]
end

function Tensor:size()
   if self._cols then
      return Size.new({ self._rows, self._cols })
   else
      return Size.new({ self._rows })
   end
end

-- 1D indexing: t[i] returns number
-- 2D indexing: t[i] returns a row proxy that supports t[i][j]
Tensor.__index = function(self, k)
   if type(k) == "number" then
      if self._cols then
         -- return a row proxy
         return self._data[k]
      else
         return self._data[k]
      end
   end
   return Tensor[k]
end

Tensor.__newindex = function(self, k, v)
   if type(k) == "number" then
      self._data[k] = v
   else
      rawset(self, k, v)
   end
end

-- index(dim, indices_tensor) - reorder along dimension
function Tensor:index(dim, indices)
   assert(dim == 1, "only dim=1 supported in mock")
   local result = {}
   for i = 1, indices._rows do
      result[i] = self._data[indices._data[i]]
   end
   local t = torch.Tensor(result)
   return t
end

------------------------------------------------------------
-- torch.sort(tensor) -> sorted_values, indices
------------------------------------------------------------
function torch.sort(tensor)
   local n = tensor._rows
   -- build index pairs
   local pairs_list = {}
   for i = 1, n do
      pairs_list[i] = { val = tensor._data[i], idx = i }
   end
   table.sort(pairs_list, function(a, b)
      if a.val == b.val then return a.idx < b.idx end
      return a.val < b.val
   end)
   local sorted_vals = {}
   local sorted_idxs = {}
   for i = 1, n do
      sorted_vals[i] = pairs_list[i].val
      sorted_idxs[i] = pairs_list[i].idx
   end
   return torch.Tensor(sorted_vals), torch.Tensor(sorted_idxs)
end

------------------------------------------------------------
-- torch.eq(tensor, value) -> tensor of 0/1
------------------------------------------------------------
function torch.eq(tensor, value)
   local result = {}
   for i = 1, tensor._rows do
      result[i] = (tensor._data[i] == value) and 1 or 0
   end
   return torch.Tensor(result)
end

------------------------------------------------------------
-- torch.sum(tensor) -> number
------------------------------------------------------------
function torch.sum(tensor)
   local s = 0
   for i = 1, tensor._rows do
      s = s + tensor._data[i]
   end
   return s
end

return torch
