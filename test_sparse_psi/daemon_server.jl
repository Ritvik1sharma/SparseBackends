# Persistent Julia worker. Start once with:
#   julia --project=. test_sparse_psi/daemon_server.jl &
# Then run tests via:
#   julia --project=. -e 'using DaemonMode; runargs()' test_sparse_psi/test_profile_bareh.jl --N-plaq 4 --eignv false ...
# All subsequent runs skip precompile.
using DaemonMode
serve(3000; print_stack=true, async=false)
