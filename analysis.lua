--- file to perform some analysis
manifold = require 'manifold'
-- stats = torch.load(arg[1])

-- vec = stats.embeddings

local file = io.open(arg[1])

vec = {}


split = function(s, pattern, maxsplit)
  local pattern = pattern or ' '
  local maxsplit = maxsplit or -1
  local s = s
  local t = {}
  local patsz = #pattern
  while maxsplit ~= 0 do
    local curpos = 1
    local found = string.find(s, pattern)
    if found ~= nil then
      table.insert(t, string.sub(s, curpos, found - 1))
      curpos = found + patsz
      s = string.sub(s, curpos)
    else
      table.insert(t, string.sub(s, curpos))
      break
    end
    maxsplit = maxsplit - 1
    if maxsplit == 0 then
      table.insert(t, string.sub(s, curpos - patsz - 1))
    end
  end
  return t
end


if file then
	i = 1
	word = ""
    for line in file:lines() do
    	-- print(line)
    	-- print("New Line")
    	if i % 2 == 1 then
    		word = line
    	else
    		-- print(line)
	    	count = 1;
	    	a = torch.Tensor(100)
	    	for i, ele in ipairs(split(line, ' ')) do
	    		-- print("Got in")
	        -- local name, address, email = unpack(line:split(" ")) --unpack turns a table like the one given (if you use the recommended version) into a bunch of separate variables
	        -- --do something with that data
	        	a[count] = (tonumber(ele) + tonumber(ele)) / 2.0
	        	-- print(count)
	        	count = count + 1
	       	end
	       	-- print(line)
	       	-- print("I am here")
	       	vec[word] = a
	    end
	    i = i + 1
    end
else
end

--normalize
for i, val in pairs(vec) do
	local norm = vec[i]:norm()
	if norm > 0 then
		vec[i]:div(norm)
	end
end

function dot(a, b)
	return torch.dot(vec[a], vec[b])
end

function nearest_neighbors()
	for i, v in pairs(vec) do
		local maxDot = -10
		local NN = i
		for j, w in pairs(vec) do
			if j ~= i then
				if torch.dot(v,w) > maxDot then
					maxDot = torch.dot(v,w)
					NN = j
				end
			end
		end
		print(i, NN ,maxDot)
	end
end

function find_len(table)
	local cnt = 0
	for k, v in pairs(table) do
		cnt = cnt+1
	end
	return cnt
end

function plot_tsne(vec)
	local n = find_len(vec)
	print(vec['you']:size(1))
	local m = torch.zeros(n-1, vec['you']:size(1))
	local i = 1
	local symbols = {}
	for k, val in pairs(vec) do
		if k~='NULL' then
			m[i] = vec[k]
			symbols[i] = k
			i = i+1
		end
	end
  opts = {ndims = 2, perplexity = 50, pca = 50, use_bh = false}
  mapped_x1 = manifold.embedding.tsne(m)
  return mapped_x1, symbols
end

tsne, symbols = plot_tsne(vec)
--write
local file = io.open('tsne.txt', "w");
for i=1, #symbols do
	file:write(symbols[i] .. ' ' .. tsne[i][1]  .. ' ' .. tsne[i][2] .. '\n')
end





