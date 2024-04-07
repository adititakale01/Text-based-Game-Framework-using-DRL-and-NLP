-- Layer to create quests and act as middle-man between Evennia and Agent
zmq = require 'lzmq'
require 'utils'
require 'torch'
local underscore = require 'underscore'
local DEBUG = false
STATE_DIM = 100
RECURRENT = 1
local DEFAULT_REWARD = -0.01
local JUNK_CMD_REWARD = -0.1
local STEP_COUNT = 0 -- count the number of steps in current episode
local QUEST_LEVELS = 1
local MAX_STEPS = 30 --has to be changed

--Simple quests
quests = {'You are hungry.','You are sleepy.', 'You are bored.', 'You are getting fat.'}
quests_mislead = {'You are not hungry.','You are not sleepy.', 'You are not bored.', 'You are not getting fat.'}

quest_actions = {'eat', 'sleep', 'watch' ,'exercise'} -- aligned to quests above
quest_checklist = {}
mislead_quest_checklist = {}
rooms = {'Living', 'Garden', 'Kitchen','Bedroom'}

actions = {"eat", "sleep", "watch", "exercise", "go"}
objects = {'north','south','east','west'} -- read rest from build file
-- order in build file: tv, bike, apple, bed

extra_vocab = {'not','but', 'now'} -- words that are necessary for initial vocab building but not in other text
symbols = {}
symbol_mapping = {}

NUM_ROOMS = 4

local current_room_description = ""



-- 	getActions = getActions,
-- 	getObjects = getObjects,
-- 	step = step_game,
-- 	newGame = newGame,
-- 	vector_function = vector_function

function mysplit(inputstr, sep)
	if sep == nil then
		sep = "%s"
	end
	local t={} ; i=1
	for str in string.gmatch(inputstr, "([^"..sep.."]+)") do
		t[i] = str
		i = i + 1
	end
	return t
end

function concatString(input)
	local length = #input
	--print(length[1])

	output = tostring(input[1])
	i = 2
	--print(i)
	while (i <= length[1])
	do
		output = output .. " " .. tostring(input[i])
		i = i+1
	end
	return output
end

function createStateMsg(vector, reward, terminal)
	stateString = concatString(vector)
	rewardString = tostring(reward)
	terminalString = tostring(terminal)
	msg = stateString .. "#" .. rewardString .. "#" .. terminalString
	return msg
end

function interact(counter)
	ctx = zmq.init(1)
	socket = ctx:socket(zmq.REP)
	socket:bind('tcp://127.0.0.1:1234' .. tostring(counter))
	print("binding succesfully")

	ctxGame = zmq.init(5)
	socketGame = ctxGame:socket(zmq.REP)
	socketGame:bind('tcp://127.0.0.1:5678' .. tostring(counter))
	gen = socketGame:recv()
	print(gen)

	available_objects = torch.Tensor(getObjectsnumber())
	for i = 1, getObjectsnumber()
	do
		available_objects[i] = i - 1
	end
	available_objects_string = concatString(available_objects)
	while(1)
	do
		--print("I came in while")
		input = socket:recv()
		--
		--print(input)
		t = mysplit(input, "#")
		s = t[1]
		if s == "getActions" then
			socket:send(tostring(getActionsnumber()))
		elseif s == "getObjects" then
			socket:send(tostring(getObjectsnumber()))
		elseif s == "step_game" then
			vector, reward, terminal = step_game(tonumber(t[2]), tonumber(t[3]))
			--print("available objects")
			--print(available_objects) @todo: have to implement available objects inorder to play fantasy world
			msg = createStateMsg(vector, reward, terminal) .. "#" .. available_objects_string
			socket:send(msg)
		elseif s == "newGame" then
			vector, reward, terminal = newGame()
			msg = createStateMsg(vector, reward, terminal, available_objects) .. "#" .. available_objects_string
			socket:send(msg)
		elseif s == "vocab_size" then
			socket:send(tostring(#symbols))
		elseif s == "vector_function" then
			socket:send("Error!!!")
		end
	end
	socket:close() --@Sagar Is this necessary
	ctx:term()	--@Sagar Is this necessary
end


function random_teleport()
	local room_index = torch.random(1, NUM_ROOMS)
	-- data_out('@tel tut#0'..room_index)
	socketGame:send("tel "..(room_index - 1))
	sleep(0.1)
	-- data_in()
	socketGame:recv()
	-- data_out('l')
	socketGame:send("look")
	if DEBUG then
		print('Start Room : ' .. room_index ..' ' .. rooms[room_index])
	end
end

function get_quest_text(quest_num)
	return quests_mislead[mislead_quest_checklist[1]] .. ' now but ' .. quests[quest_num] ..' now.' --randomized complex quests
end


function random_quest()
	indxs = torch.randperm(#quests)
	for i=1,QUEST_LEVELS do
		local quest_index = indxs[i]
		quest_checklist[#quest_checklist+1] = quest_index
	end

	--misleading quests
	mislead_quest_checklist[1] = indxs[#indxs]
	for i=1, #quest_checklist-1 do
		mislead_quest_checklist[i+1] = indxs[i]
	end

	if DEBUG then
		print("Start quest", get_quest_text(quest_checklist[1]), quest_actions[quest_checklist[1]])
	end
end

function login(user, password)
	local num_rooms = 4
	-- local pre_login_text = data_in()
	-- print("pre_login_text: ")
	-- print(pre_login_text)
	sleep(1)
	-- data_out('connect ' .. user .. ' ' .. password)
end

--Function to parse the output of the game (to extract rewards, etc. )
function parse_game_output(text)
	-- extract REWARD if it exists
	-- text is a list of sentences
	local reward = nil
	local text_to_agent = {current_room_description, get_quest_text(quest_checklist[1])}
	local sep = "\n"
	print(text)
	for str in string.gmatch(text, "([^"..sep.."]+)") do
		
		if string.match(str, "REWARD") then
			if string.match(str, quest_actions[quest_checklist[1]]) then
				reward = tonumber(string.match(str, "%d+"))
			end
		elseif string.match(str, 'not available') or string.match(str, 'not find') then
				reward = JUNK_CMD_REWARD
		else
			text_to_agent = {current_room_description, get_quest_text(quest_checklist[1])}
		end
	end
	-- for i=1, #text do
	-- 	if i < #text  and string.match(text[i], '<EOM>') then
	-- 		text_to_agent = {current_room_description, get_quest_text(quest_checklist[1])}
	-- 	elseif string.match(text[i], "REWARD") then
	-- 		if string.match(text[i], quest_actions[quest_checklist[1]]) then
	-- 			reward = tonumber(string.match(text[i], "%d+"))
	-- 		end
	-- 	elseif string.match(text[i], 'not available') or string.match(text[i], 'not find') then
	-- 			reward = JUNK_CMD_REWARD
	-- 	end
	-- end
	if not reward then
		reward = DEFAULT_REWARD
	end
	--print(text_to_agent)
	return text_to_agent, reward
end


--take a step in the game
function step_game(action_index, object_index, gameLogger)
	local command = build_command(actions[action_index], objects[object_index], gameLogger)
	-- data_out(command)
	socketGame:send(command)
	if DEBUG then
		print(actions[action_index] .. ' ' .. objects[object_index])
	end
	STEP_COUNT = STEP_COUNT + 1
	--print(getState(gameLogger))
	return getState(gameLogger)
end

-- starts a new game
function newGame(gameLogger)

	quest_checklist = {}
	mislead_quest_checklist = {}
	STEP_COUNT = 0
	random_teleport()
	random_quest()
	print(quest_checklist[1])
	if gameLogger then
	end

	return getState(gameLogger)
end

-- build game command to send to the game
function build_command(action, object, logger)
	if logger then
		logger:write(">>" .. action .. ' '.. object..'\n')
	end
	return action .. ' ' ..object
end


function parseLine( list_words, start_index)
	-- parse line to update symbols and symbol_mapping
	-- IMP: make sure we're using simple english - ignores punctuation, etc.
	local sindx
	start_index = start_index or 1
	for i=start_index,#list_words do
		word = split(list_words[i], "%a+")[1]
		word = word:lower()
		if symbol_mapping[word] == nil then
			sindx = #symbols + 1
			symbols[sindx] = word
			symbol_mapping[word] = sindx
		end
	end
end

function addQuestWordsToVocab()
	for i, quest in pairs(quests) do
		parseLine(split(quest, "%a+"), 1)
	end
end

function addExtraWordsToVocab()
	for i, word in pairs(extra_vocab) do
		word = word:lower()
		if symbol_mapping[word] == nil then
			sindx = #symbols + 1
			symbols[sindx] = word
			symbol_mapping[word] = sindx
		end
	end
end

-- read in text data from file with sentences (one sentence per line) - nicely tokenized
function makeSymbolMapping(filename)
	local file = io.open(filename, "r");
	local data = {}
	local parts
	for line in file:lines() do
		list_words = split(line, "%S+")
		if list_words[1] == '@detail' or list_words[1] == '@desc' then
			parseLine(list_words, 4)
		elseif list_words[1] == '@create/drop' then
			-- add to actionable objects
			table.insert(objects, split(list_words[2], "%a+")[1])
		end
	end
	addQuestWordsToVocab()
	addExtraWordsToVocab()
end

function writeSymbolMapping(num)
	--print("I am in text to ordered")
	-- print(symbol_mapping)
	file = io.open("symbolMapping" .. tostring(num) ..".txt", "w")
	for key, value in pairs(symbol_mapping) do
		file:write(key, " ", value, "\n")
	end
	io.close(file)
end

-- Args: {
--	1: desc of room
--	2: quest desc
-- }
function convert_text_to_bow(input_text)
	--print("convert_text_to_bow")
	-- print(input_text)
	local vector = torch.zeros(#symbols)
	for j, line in pairs(input_text) do
		line = input_text[j]
		local list_words = split(line, "%a+")
		for i=1,#list_words do
			local word = list_words[i]
			word = word:lower()
			--ignore words not in vocab
			if symbol_mapping[word] then
				vector[symbol_mapping[word]] = vector[symbol_mapping[word]] + 1
			else
				print(word .. ' not in vocab')
			end

		end
	end
	return vector
end

-- Args: {
--	1: desc of room
--	2: quest desc
-- }
function convert_text_to_bigram(input_text)
	local vector = torch.zeros(#symbols*#symbols)
	for j, line in pairs(input_text) do
		line = input_text[j]
		local list_words = split(line, "%a+")
		for i=1,#list_words-1 do
			local word = list_words[i]
			word = word:lower()
			next_word = list_words[i+1]:lower()
			--ignore words not in vocab
			if symbol_mapping[word] and symbol_mapping[next_word] then
				vector[(symbol_mapping[word]-1)* (#symbols) + symbol_mapping[next_word]]
					= vector[(symbol_mapping[word]-1)* (#symbols) + symbol_mapping[next_word]]  + 1
			else
				print(word .. ' not in vocab')
			end

		end
	end
	return vector
end

-- Args: {
--	1: desc of room
--	2: quest desc
-- }
-- Create separate bow vectors for the two sentences
function convert_text_to_bow2(input_text)
	local vector = torch.zeros(2 * #symbols)
	for j=1, 2 do
		line = input_text[j]
		local list_words = split(line, "%a+")
		for i=1,#list_words do
			local word = list_words[i]
			word = word:lower()
			--ignore words not in vocab
			if symbol_mapping[word] then
				vector[(j-1)*(#symbols) + symbol_mapping[word]]
						= vector[(j-1)*(#symbols) + symbol_mapping[word]] + 1
			else
				print(word .. ' not in vocab')
			end

		end
	end
	return vector
end


-- for recurrent networks
-- assumes that the symbol mapping has already been created
-- STATE_DIM = max desc/quest length
function convert_text_to_ordered_list(input_text)
	--local NULL_INDEX = #symbols + 1
	local NULL_INDEX = 0 -- as python indices start from 0 we can accomodate NULL at 0 instead of #symbols + 1
	--print(STATE_DIM)
	local vector = torch.ones(STATE_DIM) * NULL_INDEX
	local REVERSE = true --reverse the order of words to have padding in beginning
	cnt=1


	for j=1, 2 do
		line = input_text[j]
		local list_words = split(line, "%a+")
		for i=1,#list_words do
			local word = list_words[i]
			word = word:lower()
			if REVERSE then cnt2 = STATE_DIM+1-cnt else cnt2 = cnt end
			--ignore words not in vocab
			if symbol_mapping[word] then
				vector[cnt2] = symbol_mapping[word]
			else
				print(word .. ' not in vocab')
			end
			cnt=cnt+1
		end
	end
	-- return reverse_tensor(vector)
	return vector
end

-- for recurrent networks
-- Separate lists for description and quest.
-- STATE_DIM = max desc/quest length
function convert_text_to_ordered_list2(input_text)
	local NULL_INDEX = #symbols + 1
	local vector = torch.ones(2 * STATE_DIM) * NULL_INDEX
	local REVERSE = true --reverse the order of words to have padding in beginning
	for j=1, 2 do
		cnt=1
		line = input_text[j]
		local list_words = split(line, "%a+")
		for i=1,#list_words do
			local word = list_words[i]
			word = word:lower()
			if REVERSE then cnt2 = STATE_DIM+1-cnt else cnt2 = cnt end
			--ignore words not in vocab
			if symbol_mapping[word] then
				vector[(j-1)*STATE_DIM + cnt2] = symbol_mapping[word]
			else
				print(word .. ' not in vocab')
			end
			cnt=cnt+1
		end
	end
	-- return reverse_tensor(vector)
	return vector
end


-------------------------VECTOR function -------------------------
if RECURRENT == 1 then
	print (RECURRENT)
	vector_function = convert_text_to_ordered_list
elseif BIGRAM then
	print (RECURRENT)
	vector_function = convert_text_to_bigram
else
	print (RECURRENT)
	vector_function = convert_text_to_bow
end
-------------------------------------------------------------------

function mysplit(inputstr, sep)
    if sep == nil then
        sep = "%s"
    end
    local t={} ; i=1
    for str in string.gmatch(inputstr, "([^"..sep.."]+)") do
        t[i] = str
        i = i + 1
    end
    return t
end


function getState(logger, print_on)
	local terminal = (STEP_COUNT >= MAX_STEPS)
	-- local inData = data_in()
	local response1 = socketGame:recv()
	print(response1)
	local inData = mysplit(response1, "\n")
	-- while #response == 0 or not string.match(inData[#inData],'<EOM>') do
	-- 	TableConcat(inData, data_in())
	-- end

	-- data_out('look')
	socketGame:send("look")
	local response2 = socketGame:recv()
	local inData2 = mysplit(response2, "\n")
	-- while #inData2 == 0 or not string.match(inData2[#inData2],'<EOM>') do
	-- 	TableConcat(inData2, data_in())
	-- end
	current_room_description = inData2[1]

	local text, reward = parse_game_output(response1)
	if DEBUG or print_on then
		print(text, reward)
		sleep(0.1)
		if reward > 0 then
			print(text, reward)
			sleep(2)
		end
	end
	if reward >= 1 then
		quest_checklist = underscore.rest(quest_checklist) --remove first element in table
		mislead_quest_checklist = underscore.rest(mislead_quest_checklist) --remove first element in table
		if #quest_checklist == 0 then
			--quest has been succesfully finished
			terminal = true
		else
			text[2] = get_quest_text(quest_checklist[1])
		end
	end

	local vector = vector_function(text)

	if logger then
		logger:write(table.concat(text, ' '), '\n')
		logger:write('Reward: '..reward, '\n')
		if terminal then
			logger:write('****************************\n\n')
		end
	end
	return vector, reward, terminal
end

function getActions()
	return actions
end

function getObjects()
	return objects
end

local numactions
local numobjects

function getActionsnumber()
	if (numactions == nil) then
		numactions = #actions
	end
	return numactions
end

function getObjectsnumber()
	if (numobjects==nil) then
		numobjects = #objects
	end
	return numobjects
end
return {
	makeSymbolMapping = makeSymbolMapping,
	getActions = getActions,
	getObjects = getObjects,
	getState = getState,
	step = step_game,
	newGame = newGame,
	nextRandomGame = nextRandomGame,
	vector_function = vector_function,
	interact = interact,
	writeSymbolMapping = writeSymbolMapping
}

