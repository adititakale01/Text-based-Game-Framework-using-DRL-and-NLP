framework = require 'framework'
require 'client'
require 'utils'




if arg[1] ~= '6' then
    local port = 4000 + tonumber(arg[1])
    print(port)
    client_connect(port)
    login('root', 'root')
    counter = tonumber(arg[1])    
    framework.makeSymbolMapping("../text-world/evennia/contrib/text_sims/build.ev")
    framework.makeSymbolMapping("../text-world/evennia/contrib/text_sims3/build.ev")
    framework.makeSymbolMapping("../text-world/evennia/contrib/text_sims4/build.ev")
    framework.makeSymbolMapping("../text-world/evennia/contrib/text_sims6/build.ev")
    framework.writeSymbolMapping("1236")
	print("#symbols", #symbols)
	framework.interact(counter)    
elseif arg[1] =='6' then
    framework.makeSymbolMapping("../text-world/evennia/contrib/text_sims/build.ev")
    framework.makeSymbolMapping("../text-world/evennia/contrib/text_sims3/build.ev")
    framework.makeSymbolMapping("../text-world/evennia/contrib/text_sims4/build.ev")
    framework.makeSymbolMapping("../text-world/evennia/contrib/text_sims6/build.ev")
    framework.writeSymbolMapping("1236")
	print("#symbols", #symbols)    
end