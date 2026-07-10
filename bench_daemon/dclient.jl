# DaemonMode client: send the Julia expr in ARGS[1] (a file path) to the daemon.
# The daemon reuses already-loaded/compiled ITensors + SparseBackends, so we pay
# the package-load / JIT cost only once (at daemon warmup), not per test.
using DaemonMode
port = parse(Int, get(ENV, "DAEMON_PORT", "3000"))
runexpr(read(ARGS[1], String); port = port)
