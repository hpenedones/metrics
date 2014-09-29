local roc = {}

function roc.curve(responses, labels)

	-- put assertions here

   	-- assuming labels {-1, 1}
   	local npositives = torch.sum(torch.eq(labels,  1))
   	local nnegatives = torch.sum(torch.eq(labels, -1))
   	local nsamples = npositives + nnegatives

   	-- sort by response value
   	local responses_sorted, indexes_sorted = torch.sort(responses)
   	local labels_sorted = labels:index(1, indexes_sorted)


   	local roc_points = {}
   	roc_points[1] = {1.0, 1.0}

   	local npoints = 1
	local true_negatives = 0
	local false_negatives = 0   	
   	local i = 1

   	while i<nsamples do
		local split = responses_sorted[i]
		-- if samples have exactly the same response, can't distinguish
		-- between them with a threshold in the middle
		while i <= nsamples and responses_sorted[i] == split do
			if labels_sorted[i] == -1 then
				true_negatives = true_negatives + 1
			else
				false_negatives = false_negatives + 1
			end
			i = i+1
		end
		while i <= nsamples and labels_sorted[i] == -1 do
			true_negatives = true_negatives + 1
			i = i+1	
		end
		npoints = npoints + 1
		local false_positives = nnegatives - true_negatives
		local true_positives = npositives - false_negatives 
		local false_positive_rate = 1.0*false_positives/nnegatives
		local true_positive_rate = 1.0*true_positives/npositives
		roc_points[npoints] = { false_positive_rate, true_positive_rate }
   	end

   	npoints = npoints + 1
   	roc_points[npoints] = {0.0, 0.0}

   	local roc_tensor2d = torch.Tensor(npoints, 2)
   	for i=1,npoints do
   		roc_tensor2d[i][1] = roc_points[npoints-i+1][1]
   		roc_tensor2d[i][2] = roc_points[npoints-i+1][2]
   	end

   	return roc_tensor2d
end


function roc.area(roc_points)

	local area = 0.0 
	local npoints = roc_points:size()[1]

	for i=1,npoints-1 do
		local width = (roc_points[i+1][1] - roc_points[i][1])
		local avg_height = (roc_points[i][2]+roc_points[i+1][2])/2.0
		area = area + width*avg_height
	end

	return area
end

   
return roc