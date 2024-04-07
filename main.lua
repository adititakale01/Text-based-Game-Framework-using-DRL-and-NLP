framework = require 'framework'
require 'client'
require 'utils'



-- local port = 4000 + tonumber(arg[1])%4
-- print(port)
-- client_connect(port)
-- login('root', 'root')

counter = tonumber(arg[1])%4
if arg[1]=='1' then
    framework.makeSymbolMapping("../text-world/evennia/contrib/text_sims/build.ev")
elseif arg[1]=='2' then
    framework.makeSymbolMapping("../text-world/evennia/contrib/text_sims3/build.ev")
elseif arg[1]=='3' then
    framework.makeSymbolMapping("../text-world/evennia/contrib/text_sims5/build.ev")
elseif arg[1]=='5' then
    framework.makeSymbolMapping("../text-world/evennia/contrib/text_sims/build.ev")
    framework.makeSymbolMapping("../text-world/evennia/contrib/text_sims3/build.ev")
    framework.makeSymbolMapping("../text-world/evennia/contrib/text_sims4/build.ev")
end

framework.writeSymbolMapping(arg[1])
print("#symbols", #symbols)
framework.interact(counter)
