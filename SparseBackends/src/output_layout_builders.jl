# ─────────────────────────────────────────────────────────────────────────────
# OutputLayout builders for the known callers, and role resolution (resolve_layout,
# show_roles). (Included after tensor_wrappers.jl, tensor_contraction.jl,
# tensor_wrappers_aliased.jl.)
# ─────────────────────────────────────────────────────────────────────────────

"""
    dense_from_aliased_operand(Aw, Bw) -> OutputLayout

Layout used by the DMRG environment/matvec products (`_mul_preserve_aliased`) and
`mul_aliased`: the dense block holds only the aliased operand(s)' dense indices; every
other output index (e.g. the dense operand's site/link indices) goes in the key.
An output index is dense iff its id is one of those dense indices (matched by id,
ignoring prime level).
"""
function dense_from_aliased_operand(Aw::WrappedTensorTypes, Bw::WrappedTensorTypes)
    a_ali = Aw isa WrappedAliasedBlockSparse
    b_ali = Bw isa WrappedAliasedBlockSparse
    (a_ali || b_ali) || throw(OutputLayoutError(
        "dense_from_aliased_operand: neither operand has aliased storage"))
    hint = Set{ITensors.Index}()
    a_ali && union!(hint, dense_inds(Aw))
    b_ali && union!(hint, dense_inds(Bw))
    indsC, denseC = output_inds(Aw.inds, Bw.inds, dense_inds(Aw), dense_inds(Bw);
                                dense_ids=hint)
    return _layout_from_output_inds(indsC, denseC)
end

"""
    _absorb_layout(inds, P, old, new) -> OutputLayout

Layout of M * R for the per-storage `_absorb_bond` methods: M's own index order and
key/dense split (`inds` = M's stored order, the first `P` in the key, the rest dense), with
R's new bond index `new[i]` in the slot of `old[i]` (both in R's stored order).
"""
function _absorb_layout(inds::NTuple{N,ITensors.Index}, P::Int, old, new) where {N}
    sparse = Vector{LayoutEntry}(undef, P)
    dense  = Vector{LayoutEntry}(undef, N - P)
    for j in 1:N
        I = inds[j]
        for i in eachindex(old)
            if old[i] == I
                I = new[i]
                break
            end
        end
        j <= P ? (sparse[j] = I) : (dense[j - P] = I)
    end
    return OutputLayout(sparse, dense)
end

# ── roles (see the vocabulary in output_layout.jl) ───────────────────────────

function _role_num(I, key::String)
    for t in ITensors.tags(I)
        m = match(Regex("^" * key * raw"=(\d+)$"), string(t))
        m === nothing || return parse(Int, m.captures[1])
    end
    return nothing
end
_role_issite(I) = ITensors.hastags(I, "Site")
_role_islink(I) = ITensors.hastags(I, "Link")

# kind of each index of `inds` (:l, :r, :s, :sp, or nothing), from the site numbers n=
# and the site prime levels of this operand (:s = lowest, :sp = highest)
function _role_kinds(inds)
    ns = [_role_num(I, "n") for I in inds if _role_issite(I)]
    ns = filter(!isnothing, ns)
    nmin, nmax = isempty(ns) ? (nothing, nothing) : (minimum(ns), maximum(ns))
    pls = [ITensors.plev(I) for I in inds if _role_issite(I)]
    plo, phi = isempty(pls) ? (0, 0) : (minimum(pls), maximum(pls))
    map(inds) do I
        if _role_issite(I)
            ITensors.plev(I) == plo ? :s : ITensors.plev(I) == phi ? :sp : nothing
        elseif _role_islink(I) && nmin !== nothing
            j = _role_num(I, "l")
            j === nothing ? nothing : j == nmin - 1 ? :l : j == nmax ? :r : nothing
        else
            nothing
        end
    end
end

# (kind, k) => Index for one operand
function _role_map(inds)
    kinds = _role_kinds(inds)
    rm = Dict{Tuple{Symbol,Int},ITensors.Index}()
    cnt = Dict{Symbol,Int}()
    for (I, kd) in zip(inds, kinds)
        kd === nothing && continue
        k = cnt[kd] = get(cnt, kd, 0) + 1
        rm[(kd, k)] = I
    end
    return rm
end

function _parse_role(r::Role)
    op, kind = r[1], r[2]
    k = length(r) == 3 ? r[3] : 1
    (op in (:A, :B) && kind in (:l, :r, :s, :sp) && k >= 1) ||
        throw(OutputLayoutError("bad role $r: expected (op, kind[, k]) with op :A or :B, " *
                                "kind :l, :r, :s or :sp, k ≥ 1 (see show_roles)"))
    return (op === :A ? 1 : 2), kind, k
end

"""
    resolve_layout(L, Aw, Bw) -> OutputLayout

Replace the roles in `L` by the Index objects they mean for this contraction (`Aw`
first operand, `Bw` second). Roles absent at this site are skipped; Index entries are
kept. The result is checked later by `validate_layout`.
"""
function resolve_layout(L::OutputLayout, Aw::WrappedTensorTypes, Bw::WrappedTensorTypes)
    isresolved(L) && return L
    sA, sB = Set(Aw.inds), Set(Bw.inds)
    outs = (Set(I for I in Aw.inds if !(I in sB)), Set(I for I in Bw.inds if !(I in sA)))
    rms  = (_role_map(collect(Aw.inds)), _role_map(collect(Bw.inds)))
    function res(list)
        got = ITensors.Index[]
        for x in list
            if x isa ITensors.Index
                push!(got, x)
            else
                o, kind, k = _parse_role(x)
                I = get(rms[o], (kind, k), nothing)
                (I === nothing || !(I in outs[o])) && continue     # absent here: skip
                push!(got, I)
            end
        end
        return got
    end
    return OutputLayout(res(L.sparse), res(L.dense))
end

"""
    show_roles(T; as=:A)

Print each index of `T` (ITensor or wrapped tensor) in stored order with its role for
an OutputLayout (`as=:A` if T is the first operand of the contraction, `:B` the second),
whether it is in the key or the dense block, dimension, prime level and tags.
"""
function show_roles(io::IO, w::WrappedTensorTypes; as::Symbol=:A)
    rm  = _role_map(collect(w.inds))
    inv = Dict(I => r for (r, I) in rm)
    d   = dense_inds(w)
    println(io, typeof(w).name.name, " with ", length(w.inds), " indices")
    for I in w.inds
        role = haskey(inv, I) ? "(:$as, :$(inv[I][1]), $(inv[I][2]))" : "(no role)"
        println(io, "  ", rpad(role, 14), rpad(I in d ? "dense" : "key", 6),
                " dim=", rpad(ITensors.dim(I), 5), " plev=", ITensors.plev(I),
                "  ", ITensors.tags(I))
    end
end
show_roles(io::IO, T::ITensors.ITensor; as::Symbol=:A) =
    show_roles(io, ITensors.has_external_storage(T) ? ITensors.get_external_storage(T) :
                   wrap_itensor(T; backend=:dense); as=as)
show_roles(T; as::Symbol=:A) = show_roles(stdout, T; as=as)

"""
    matvec_dense_layout(bondtype, step, A, B) -> Union{Nothing,OutputLayout}

Fixed, per-position output order for the DMRG dense-ψ × aliased-H matvec (steps 2 and 3;
bond type `:left`, `:bulk` or `:right`), from the precomputed table
`_STATIC_OUTPUT_PERM_DENSE` (a permutation of the stock output order). `A` is the
aliased operator site, `B` the dense intermediate, as in the matvec call. Returns
`nothing` where the table has no entry (steps 1 and 4), meaning the stock order.
Throws `OutputLayoutError` if the entry does not fit this contraction.
"""
function matvec_dense_layout(bondtype::Symbol, step::Int, A::ITensors.ITensor, B::ITensors.ITensor)
    perm = _static_output_perm_dense(bondtype, step)
    perm === nothing && return nothing
    Aw = wrap_itensor_aliased(A)
    Bw = wrap_itensor(B; backend=:dense)
    indsC, _ = output_inds(Aw.inds, Bw.inds, dense_inds(Aw), dense_inds(Bw))
    length(perm) == length(indsC) || throw(OutputLayoutError(
        "matvec_dense_layout($bondtype, $step): table entry has $(length(perm)) positions, " *
        "the output has $(length(indsC)) indices"))
    return OutputLayout(ITensors.Index[], collect(ITensors.Index, indsC)[perm])
end
