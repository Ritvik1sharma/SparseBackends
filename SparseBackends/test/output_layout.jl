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
end
