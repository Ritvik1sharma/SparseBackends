# OutputLayout: a requested layout is obeyed (checked against plain ITensors contraction,
# matched by index), the builders reproduce the old keywords exactly, and invalid
# layouts are rejected. Aliased operands are produced by a (:coo, :dense) -> :aliased
# multiplication, as in the real code.
using Test, Random, ITensors, SparseBackends
import SparseBackends: wrapped_contract_aliased, wrap_itensor, dense_inds, layout_inds

# dense ITensor of a wrapped result (aliased or already dense), in its own index order
_as_dense(C::ITensor) = C
_as_dense(Cw::SparseBackends.WrappedAliasedBlockSparse) =
    ITensor(SparseBackends.to_dense(Cw.aliased), Cw.inds...)
# compare two ITensors by index (not by position)
_same(X::ITensor, Y::ITensor; atol=1e-12) =
    Set(inds(X)) == Set(inds(Y)) && isapprox(array(X, inds(Y)...), array(Y); atol=atol)

@testset "OutputLayout" begin
    Random.seed!(7)
    s  = Index(3, "Site,n=1")
    p  = Index(2, "Link,l=1,P")       # projector link (lands in A's key)
    h  = Index(4, "Link,l=1,H")       # operator link (lands in A's dense block)
    x  = Index(3, "Link,x")           # free index of the dense operand B
    t  = Index(3, "Site,n=2")         # H's output site: its own id, so no key index shares an
                                      # id with a dense one (the old hint matches by id only)
    # sparse projector-like P(s, s', p) and dense H(s', t, h)
    Pa = zeros(3, 3, 2); Pa[1,1,1] = 1; Pa[2,2,2] = 1; Pa[3,3,1] = 1; Pa[2,1,2] = 1
    P  = ITensor(Pa, s, s', p)
    H  = random_itensor(s', t, h)
    A  = SparseBackends.contract(P, H, :coo, :dense, :aliased)    # aliased: key (s, p), dense (t, h)
    Aw = ITensors.get_external_storage(A)
    @test Aw isa SparseBackends.WrappedAliasedBlockSparse
    B  = random_itensor(p, x)                                       # contract over p (in A's key)
    Bw = wrap_itensor(B; backend=:dense)
    Adense = ITensor(SparseBackends.to_dense(Aw.aliased), Aw.inds...)
    ref = Adense * B

    @testset "default layout matches the reference" begin
        C0 = wrapped_contract_aliased(Aw, Bw; preserve_bs_output=true)
        @test _same(_as_dense(C0), ref)
    end

    @testset "dense_from_aliased_operand == old output_inds_hint, bit for bit" begin
        hint = Set{Index}(dense_inds(Aw))
        Cold = wrapped_contract_aliased(Aw, Bw; preserve_bs_output=true, output_inds_hint=hint)
        L    = dense_from_aliased_operand(Aw, Bw)
        Cnew = wrapped_contract_aliased(Aw, Bw; preserve_bs_output=true, layout=L)
        @test Cnew.inds == Cold.inds
        @test collect(Cnew.inds) == layout_inds(L)
        @test SparseBackends.to_dense(Cnew.aliased) == SparseBackends.to_dense(Cold.aliased)
        @test _same(_as_dense(Cnew), ref)
        @test x in L.sparse            # the dense operand's free index moved into the key
    end

    @testset "a non-default layout is obeyed" begin
        L0 = SparseBackends._layout_from_output_inds(
                 SparseBackends.output_inds(Aw.inds, Bw.inds, dense_inds(Aw), dense_inds(Bw))...)
        # same split, dense block groups swapped ([B part, A part] instead of [A part, B part])
        a_part = [I for I in L0.dense if I in Set(Aw.inds)]
        b_part = [I for I in L0.dense if !(I in Set(Aw.inds))]
        L = OutputLayout(sparse=L0.sparse, dense=vcat(b_part, a_part))
        @test L.dense != L0.dense
        C = wrapped_contract_aliased(Aw, Bw; preserve_bs_output=true, layout=L)
        @test collect(C.inds) == layout_inds(L)
        @test _same(_as_dense(C), ref)
    end

    @testset "dense output in a requested order" begin
        D0 = wrapped_contract_aliased(Aw, Bw)                 # stock dense order
        order = reverse(collect(inds(D0)))
        D = wrapped_contract_aliased(Aw, Bw; layout=OutputLayout(sparse=Index[], dense=order))
        @test collect(inds(D)) == order
        @test _same(D, ref)
    end

    @testset "invalid layouts are rejected" begin
        L = dense_from_aliased_operand(Aw, Bw)
        @test_throws OutputLayoutError wrapped_contract_aliased(Aw, Bw; preserve_bs_output=true,
            layout=OutputLayout(sparse=L.sparse[2:end], dense=L.dense))                 # missing index
        @test_throws OutputLayoutError wrapped_contract_aliased(Aw, Bw; preserve_bs_output=true,
            layout=OutputLayout(sparse=vcat(L.sparse, L.sparse[1:1]), dense=L.dense))   # duplicate
        @test_throws OutputLayoutError wrapped_contract_aliased(Aw, Bw; preserve_bs_output=true,
            layout=OutputLayout(sparse=vcat(L.sparse, [Index(2, "stray")]), dense=L.dense)) # extra
        @test_throws OutputLayoutError wrapped_contract_aliased(Aw, Bw; layout=L)       # dense output with a key
        @test_throws ArgumentError wrapped_contract_aliased(Aw, Bw; preserve_bs_output=true,
            layout=L, output_inds_hint=Set{Index}(dense_inds(Aw)))                         # old + new together
        @test_throws OutputLayoutError dense_from_aliased_operand(wrap_itensor(Adense; backend=:dense), Bw)
    end

    @testset "roles" begin
        # a TEBD-like site n=2: aliased ψ = P·core (P's links in the key, core's links in
        # the dense block, both tagged l=1 / l=2) and a dense gate G(s', s, gl, gr)
        Random.seed!(11)
        z  = Index(2, "Site,n=2")
        pl = Index(2, "Link,l=1,P"); pr = Index(2, "Link,l=2,P")
        cl = Index(3, "Link,l=1,C"); cr = Index(3, "Link,l=2,C")
        gl = Index(2, "Link,l=1,G"); gr = Index(2, "Link,l=2,G")
        Pz = zeros(2, 2, 2, 2); Pz[1,1,1,1] = 1; Pz[2,2,2,1] = 1; Pz[2,1,1,2] = 1
        Ps = SparseBackends.contract(ITensor(Pz, z, z', pl, pr), random_itensor(z', cl, cr),
                                     :coo, :dense, :aliased)
        Psw = ITensors.get_external_storage(Ps)
        @test Psw isa SparseBackends.WrappedAliasedBlockSparse
        G  = random_itensor(z', z, gr, gl)
        Gw = wrap_itensor(G; backend=:dense)
        refT = ITensor(SparseBackends.to_dense(Psw.aliased), Psw.inds...) * G
        R(L) = SparseBackends.resolve_layout(L, Gw, Psw)          # A = gate, B = ψ
        # k counts in stored order, key first: (:B,:l,1) = P's link, (:B,:l,2) = core's
        Lr = OutputLayout(sparse=[(:B,:l,1), (:B,:r,1), (:A,:sp)],
                          dense =[(:B,:l,2), (:B,:r,2), (:A,:r), (:A,:l)])
        L = R(Lr)
        @test L.sparse == [pl, pr, z'] && L.dense == [cl, cr, gr, gl]
        # roles name the operands as passed (gate first, as in contract(MPO, MPS)); the
        # aliased operand is put first internally, which needs no change to the layout
        C = wrapped_contract_aliased(Gw, Psw; preserve_bs_output=true, layout=Lr)
        @test collect(C.inds) == [pl, pr, z', cl, cr, gr, gl]
        @test _same(_as_dense(C), refT)
        # (op, kind) is (op, kind, 1); Index entries pass through
        @test R(OutputLayout(sparse=[(:A,:sp), pl], dense=ITensors.Index[])).sparse == [z', pl]
        # absent (:A,:l,2), contracted (:B,:s) and out-of-range (:B,:r,3) roles are skipped
        L = R(OutputLayout(sparse=[(:B,:s), (:A,:l,2), (:B,:r,3), (:B,:l)], dense=ITensors.Index[]))
        @test L.sparse == [pl]
        # left/right come from the site number, not the l= order: on site 1 the l=1 link is r
        y  = Index(2, "Site,n=1"); y1 = Index(2, "Link,l=1")
        rm = SparseBackends._role_map([y, y1])
        @test rm[(:r, 1)] == y1 && !haskey(rm, (:l, 1))
        # the same roles with ψ passed first: now ψ is :A
        C2 = wrapped_contract_aliased(Psw, Gw; preserve_bs_output=true,
                 layout=OutputLayout(sparse=[(:A,:l,1), (:A,:r,1), (:B,:sp)],
                                     dense =[(:A,:l,2), (:A,:r,2), (:B,:r), (:B,:l)]))
        @test C2.inds == C.inds && SparseBackends.to_dense(C2.aliased) == SparseBackends.to_dense(C.aliased)
        # dense output in a role order
        D = wrapped_contract_aliased(Gw, Psw; layout=OutputLayout(sparse=ITensors.Index[],
                dense=[(:A,:sp), (:A,:l), (:A,:r), (:B,:l,1), (:B,:l,2), (:B,:r,1), (:B,:r,2)]))
        @test collect(inds(D)) == [z', gl, gr, pl, cl, pr, cr]
        @test _same(D, refT)
        # bad roles and incomplete layouts are errors
        @test_throws OutputLayoutError R(OutputLayout(sparse=[(:C,:s)], dense=ITensors.Index[]))
        @test_throws OutputLayoutError R(OutputLayout(sparse=[(:A,:x)], dense=ITensors.Index[]))
        @test_throws OutputLayoutError R(OutputLayout(sparse=[(:A,:s,0)], dense=ITensors.Index[]))
        @test_throws OutputLayoutError wrapped_contract_aliased(Gw, Psw; preserve_bs_output=true,
            layout=OutputLayout(sparse=[(:B,:l,1), (:B,:r,1)], dense=[(:B,:l,2), (:B,:r,2)]))
        # show_roles lists every index with its role and key/dense
        io = IOBuffer(); show_roles(io, Ps; as=:B); out = String(take!(io))
        @test occursin("(:B, :l, 1)", out) && occursin("(:B, :l, 2)", out) && occursin("dense", out)
    end
end
