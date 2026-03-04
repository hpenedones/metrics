-- Comprehensive tests for roc.lua (ROC line and area computation)
-- Uses a lightweight torch mock so that tests can run without the full Torch/LuaJIT stack.
--
-- Run from the repository root:
--   lua5.3 test/test_roc.lua

-- ----------------------------------------------------------------
-- Setup: load mock torch into the global namespace, then load roc
-- ----------------------------------------------------------------
package.path = package.path .. ";./test/?.lua;./?.lua"
torch = require("mock_torch")
local roc = require("roc")

local num_passed = 0
local num_failed = 0
local DEFAULT_TOL = 1e-9

local function approx_equal(a, b, tol)
   tol = tol or DEFAULT_TOL
   return math.abs(a - b) <= tol
end

local function check(condition, name)
   if condition then
      num_passed = num_passed + 1
      print("  PASS: " .. name)
   else
      num_failed = num_failed + 1
      print("  FAIL: " .. name)
   end
end

-- ================================================================
-- Test 1: Perfect classifier – all positives scored higher than all negatives
-- Expected AUC = 1.0
-- ================================================================
print("\n=== Test 1: Perfect classifier ===")
do
   local resp   = torch.DoubleTensor({ 0.1, 0.2, 0.9, 0.8 })
   local labels = torch.IntTensor({    -1,  -1,   1,   1  })

   local roc_points, thresholds = roc.points(resp, labels)
   local area = roc.area(roc_points)

   check(approx_equal(area, 1.0), "AUC should be 1.0 for perfect classifier, got " .. area)

   -- First point should be (0,0), last should be (1,1)
   check(approx_equal(roc_points[1][1], 0) and approx_equal(roc_points[1][2], 0),
         "ROC curve should start at (0,0)")
   local n = roc_points:size()[1]
   check(approx_equal(roc_points[n][1], 1) and approx_equal(roc_points[n][2], 1),
         "ROC curve should end at (1,1)")
end

-- ================================================================
-- Test 2: Worst classifier – all positives scored lower than all negatives
-- Expected AUC = 0.0
-- ================================================================
print("\n=== Test 2: Worst classifier ===")
do
   local resp   = torch.DoubleTensor({ 0.9, 0.8, 0.1, 0.2 })
   local labels = torch.IntTensor({    -1,  -1,   1,   1  })

   local roc_points, _ = roc.points(resp, labels)
   local area = roc.area(roc_points)

   check(approx_equal(area, 0.0), "AUC should be 0.0 for worst classifier, got " .. area)
end

-- ================================================================
-- Test 3: Classifier with AUC = 0.5
-- Positives at scores {0.1, 0.4}, negatives at {0.2, 0.3}.
-- Pairs: (0.1,0.2)->0, (0.1,0.3)->0, (0.4,0.2)->1, (0.4,0.3)->1  => AUC = 2/4 = 0.5
-- ================================================================
print("\n=== Test 3: Classifier with AUC = 0.5 ===")
do
   local resp   = torch.DoubleTensor({ 0.1, 0.2, 0.3, 0.4 })
   local labels = torch.IntTensor({     1,  -1,  -1,   1  })

   local roc_points, _ = roc.points(resp, labels)
   local area = roc.area(roc_points)

   check(approx_equal(area, 0.5), "AUC should be 0.5, got " .. area)
end

-- ================================================================
-- Test 4: Unsorted responses – specifically targets the bug where
-- responses[1] is used instead of responses_sorted[1] for the
-- initial threshold.  When responses are NOT already sorted the
-- initial threshold can be set too high, corrupting TN/FN counts.
-- ================================================================
print("\n=== Test 4: Unsorted responses (bug detector) ===")
do
   -- Perfect classifier but responses are in REVERSE order
   local resp   = torch.DoubleTensor({ 0.9, 0.8, 0.2, 0.1 })
   local labels = torch.IntTensor({     1,   1,  -1,  -1  })

   local roc_points, _ = roc.points(resp, labels)
   local area = roc.area(roc_points)

   check(approx_equal(area, 1.0),
         "AUC should be 1.0 for perfect classifier with reversed input order, got " .. area)
end

-- ================================================================
-- Test 5: Unsorted responses – pathological case where
-- responses[1]-epsilon equals an existing response value
-- ================================================================
print("\n=== Test 5: Pathological unsorted (responses[1]-eps == another response) ===")
do
   -- responses[1] = 0.11, so initial threshold = 0.11-0.01 = 0.10
   -- responses_sorted[1] = 0.1 which EQUALS the wrong threshold
   -- This causes the tie-handling loop to incorrectly consume
   -- the first sorted sample at the initial split.
   local resp   = torch.DoubleTensor({ 0.11, 0.1,  0.5 })
   local labels = torch.IntTensor({     1,   -1,    1  })

   local roc_points, _ = roc.points(resp, labels)
   local area = roc.area(roc_points)

   -- This is a perfect classifier; AUC must be 1.0
   check(approx_equal(area, 1.0),
         "AUC should be 1.0 for perfect classifier (pathological unsorted), got " .. area)
end

-- ================================================================
-- Test 6: Tied responses (same score for positive and negative)
-- ================================================================
print("\n=== Test 6: Tied responses ===")
do
   local resp   = torch.DoubleTensor({ 0.5, 0.5, 0.5, 0.5 })
   local labels = torch.IntTensor({    -1,   1,  -1,   1  })

   local roc_points, _ = roc.points(resp, labels)
   local area = roc.area(roc_points)

   -- When all responses are identical the classifier cannot distinguish,
   -- curve should go (0,0) -> (1,1) giving AUC = 0.5
   check(approx_equal(area, 0.5), "AUC should be 0.5 with all tied responses, got " .. area)
end

-- ================================================================
-- Test 7: Single sample per class
-- ================================================================
print("\n=== Test 7: Single sample per class ===")
do
   local resp   = torch.DoubleTensor({ 0.3, 0.7 })
   local labels = torch.IntTensor({    -1,   1  })

   local roc_points, _ = roc.points(resp, labels)
   local area = roc.area(roc_points)

   check(approx_equal(area, 1.0), "AUC should be 1.0 for 2-sample perfect classifier, got " .. area)
end

-- ================================================================
-- Test 8: Reproduce the original test from test.lua
-- (responses happen to be sorted, so the bug is hidden)
-- ================================================================
print("\n=== Test 8: Original example (sorted responses) ===")
do
   local resp   = torch.DoubleTensor({ -0.9, -0.8, -0.8, -0.5, -0.1, 0.0, 0.2, 0.2, 0.51, 0.74, 0.89 })
   local labels = torch.IntTensor({      -1,   -1,    1,   -1,   -1,   1,   1,  -1,   -1,    1,    1  })

   local roc_points, _ = roc.points(resp, labels)
   local area = roc.area(roc_points)

   check(area >= 0.7 and area <= 0.75,
         "AUC should be between 0.7 and 0.75 for the original example, got " .. area)
end

-- ================================================================
-- Test 9: Custom positive/negative labels
-- ================================================================
print("\n=== Test 9: Custom labels (0 and 1) ===")
do
   local resp   = torch.DoubleTensor({ 0.1, 0.9 })
   local labels = torch.IntTensor({     0,   1  })

   local roc_points, _ = roc.points(resp, labels, 0, 1)
   local area = roc.area(roc_points)

   check(approx_equal(area, 1.0), "AUC should be 1.0 with custom labels, got " .. area)
end

-- ================================================================
-- Test 10: ROC curve monotonicity – FPR should be non-decreasing
-- ================================================================
print("\n=== Test 10: ROC curve monotonicity ===")
do
   local resp   = torch.DoubleTensor({ 0.1, 0.4, 0.35, 0.8, 0.7, 0.6 })
   local labels = torch.IntTensor({    -1,   1,   -1,   1,  -1,   1  })

   local roc_points, _ = roc.points(resp, labels)
   local npoints = roc_points:size()[1]

   local monotonic = true
   for i = 1, npoints - 1 do
      if roc_points[i + 1][1] < roc_points[i][1] - DEFAULT_TOL then
         monotonic = false
         break
      end
   end
   check(monotonic, "FPR values should be non-decreasing along the ROC curve")
end

-- ================================================================
-- Test 11: Area should be between 0 and 1
-- ================================================================
print("\n=== Test 11: AUC bounds ===")
do
   local resp   = torch.DoubleTensor({ 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9 })
   local labels = torch.IntTensor({     1,  -1,   1,  -1,   1,  -1,   1,  -1  })

   local roc_points, _ = roc.points(resp, labels)
   local area = roc.area(roc_points)

   check(area >= 0.0 and area <= 1.0, "AUC should be between 0 and 1, got " .. area)
end

-- ================================================================
-- Test 12: Thresholds should be returned in non-increasing order
-- (the first ROC point (0,0) has the highest threshold, the last (1,1) the lowest)
-- ================================================================
print("\n=== Test 12: Thresholds ordering ===")
do
   local resp   = torch.DoubleTensor({ 0.5, 0.1, 0.9, 0.3 })
   local labels = torch.IntTensor({     1,  -1,   1,  -1  })

   local roc_points, thresholds = roc.points(resp, labels)
   local n = thresholds:size()[1]
   local ordered = true
   for i = 1, n - 1 do
      if thresholds[i + 1][1] > thresholds[i][1] + DEFAULT_TOL then
         ordered = false
         break
      end
   end
   check(ordered, "Thresholds should be in non-increasing order")
end

-- ================================================================
-- Summary
-- ================================================================
print(string.format("\n=== Results: %d passed, %d failed ===", num_passed, num_failed))
if num_failed > 0 then
   os.exit(1)
end
