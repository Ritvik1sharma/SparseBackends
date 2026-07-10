using ITensors, ITensorMPS, SparseBackends, LinearAlgebra, Printf
include("test_aliased_kl.jl")
_dense(x)=ITensors.has_external_storage(x) ? SparseBackends.to_dense_itensors_unfused(x) : x
function dd(t); a=t.tensor.data.aliased; (length(a.keys), a.n_templates, round(length(a.keys)/max(a.n_templates,1),digits=2)); end
function chk(name, psi, b)
    orthogonalize!(psi,b); A=psi[b]
    G = *(A, dag(prime(A,"Link")); preserve_bs_output=true)
    nk0,nt0,dd0 = dd(G); Gd0 = copy(_dense(G))
    SparseBackends.compress_aliased_templates!(G)
    nk1,nt1,dd1 = dd(G); err = norm(_dense(G)-Gd0)/max(norm(Gd0),eps())
    @printf("[%s b=%d] before: nk=%d nt=%d dedup=%.2f  →  after compress: nk=%d nt=%d dedup=%.2f   value-err=%.2e\n",
            name,b, nk0,nt0,dd0, nk1,nt1,dd1, err)
end
Hk,pk=build_setup(6,-1,3); chk("KL",pk,6); chk("KL",pk,7)
