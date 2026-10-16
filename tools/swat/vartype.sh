#!/bin/sh
# Print the C++ declaration of SWAT+ module variables (or type components) by name.
H="$(dirname "$0")/../../src/cpp/swat/modules.hpp"
for n in "$@"; do grep -n -E "^    [A-Za-z_:<>, 0-9]+ $n( =|;|\{)" "$H" | head -3; done
