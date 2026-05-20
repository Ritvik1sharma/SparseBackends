#!/bin/bash
# Run a script through the daemon. Usage: ./d.sh test_sparse_psi/test_profile_bareh.jl --N-plaq 4 --eignv false
cd /home/ritvik/temp/temp/edited_packages
julia --project=. -e 'using DaemonMode; runargs()' "$@"
