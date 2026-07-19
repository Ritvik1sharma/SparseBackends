# matvec_kernel_path_test.jl — verify the matvec KERNEL PATH (not just the value).
# Spec: Lenv·φ, ·PH[b], ·PH[b+1] must stay ALIASED (dedup preserved); the terminal
# ·Renv must produce the channel-free DENSE core'. Prints storage type + dedup
# ratio (n_keys/n_templates) at each step so we SEE the dedup survive the chain.

using ITensors, ITensorMPS, SparseBackends, Printf, Random, LinearAlgebra

ITensors.op(::OpName"Xp", ::SiteType"S=1") = [0 0 1 ; 0 0 0 ; 1 0 0]
ITensors.op(::OpName"Px", ::SiteType"S=1") = [0 1 0 ; 1 0 0 ; 0 0 0]
ITensors.op(::OpName"LP", ::SiteType"S=1") = [1 0 0 ; 0 1 0 ; 0 0 0]
ITensors.op(::OpName"RP", ::SiteType"S=1") = [1 0 0 ; 0 0 0 ; 0 0 1]
_nz(dims, coords) = (A=zeros(Float64,dims...); for c in coords; A[(c .+ 1)...]=1.0; end; A)
const Rf = _nz((3,3,2),[(0,0,0),(1,1,1),(2,2,0)])
const Rb = _nz((3,3,2,2),[(0,0,0,0),(0,0,1,0),(1,1,0,1),(1,1,1,0),(2,2,0,0)])
const Rl = _nz((3,3,2),[(0,0,0),(0,0,1),(1,1,0),(1,1,1),(2,2,0)])
function NotEqlsLoop_R1(sites)
    N=length(sites); bonds=[Index(2,"Link,l=$i") for i in 1:N-1]; W=Vector{ITensor}(undef,N)
    W[1]=ITensor(Rf, sites[1], sites[1]', bonds[1])
    for j in 2:N-1; W[j]=ITensor(Rb, sites[j], sites[j]', bonds[j-1], bonds[j]); end
    W[N]=ITensor(Rl, sites[N], sites[N]', bonds[N-1]); MPO(W)
end
function build_setup_pxp(N; sw=0, H=nothing, psi=nothing)
    Random.seed!(42); sites=siteinds("S=1",N)
    HT=OpSum(); HT+=1,"Xp",1
    for j in 0:N-2; HT+=1,"Px",j+1,"LP",j+2; HT+=1,"RP",j+1,"Xp",j+2; end
    HT+=1,"Px",N; H=MPO(HT,sites)
    P=NotEqlsLoop_R1(sites); psi0=random_mps(sites)
    psi=replaceprime(contract(P,copy(psi0),:coo,:aliased;denseLinksB=0),1=>0)
    if sw > 0
        s=Sweeps(sw); setmaxdim!(s,40); setmindim!(s,40); setcutoff!(s,1e-10)
        _,psi,_,_ = dmrg(H, psi, s; outputlevel=0, use_early_exit=false,
                         run_mode=:bop_aliased, minv_from_p=nothing, gram_from_h=false)
    end
    return H, psi, P
end

function report(tag, T)
    if ITensors.has_external_storage(T)
        w = ITensors.get_external_storage(T)
        if w isa SparseBackends.WrappedAliasedBlockSparse
            a = w.aliased; nk = length(a.keys); nt = a.n_templates
            @printf("  %-12s ALIASED   n_keys=%-4d n_templates=%-4d  dedup=%.2fx  blksize=%d\n",
                    tag, nk, nt, nk/max(nt,1), a.blksize)
            return
        elseif w isa SparseBackends.WrappedBlockSparse
            @printf("  %-12s BLOCKSPARSE\n", tag); return
        end
    end
    @printf("  %-12s DENSE     order=%d  inds=%s\n", tag, order(T),
            join([@sprintf("d%d", dim(i)) for i in inds(T)], "×"))
end

for sw in (0, 4)
    N = 12; b = div(N, 2)
    H, psi, P = build_setup_pxp(N; sw=sw)
    ITensorMPS.orthogonalize!(psi, b)
    PHa = ProjMPO(H); ITensorMPS.set_nsite!(PHa, 2); ITensorMPS.position!(PHa, psi, b)
    Lenv = ITensorMPS.lproj(PHa); Renv = ITensorMPS.rproj(PHa)
    PH = ITensorMPS.build_ph_output(P, H)

    println("\n===== matvec kernel path  (sw=$sw, bond b=$b) =====")
    report("Lenv", Lenv); report("Renv", Renv)
    phi = *(psi[b], psi[b+1]; preserve_bs_output=true)
    report("φ", phi)
    t1 = Lenv * phi;      report("Lenv·φ", t1)
    t2 = t1 * PH[b];      report("·PH[b]", t2)
    t3 = t2 * PH[b+1];    report("·PH[b+1]", t3)
    v  = t3 * Renv;       report("·Renv", v)
end
nothing
