"""
    seqd = DiscreteEddySequence(Gx, Gy, Gz, B1, Δf, ψ, ADC, excitation_bool, t, Δt, Ez2, Ezx, Ezy, Exy, Ex2y2)

A sampled version of a Sequence struct, containing ideal event amplitudes alongside computed 2nd-order spatial eddy current fields at specified times.

# Arguments
- `Gx`: (`::AbstractVector{T<:Real}`, `[T/m]`) x-gradient vector 
- `Gy`: (`::AbstractVector{T<:Real}`, `[T/m]`) y-gradient vector 
- `Gz`: (`::AbstractVector{T<:Real}`, `[T/m]`) z-gradient vector 
- `B1`: (`::AbstractVector{Complex{T<:Real}}`, `[T]`) RF amplitude vector 
- `Δf`: (`::AbstractVector{T<:Real}`, `[Hz]`) RF carrier frequency displacement vector 
- `ψ`: (`::AbstractVector{T<:Real}`, `[rad]`) RF rotating-frame phase vector 
- `ADC`: (`::AbstractVector{Bool}`) ADC sample vector 
- `excitation_bool`: (`::AbstractVector{Bool}`) RF excitation interval vector 
- `t`: (`::AbstractVector{T<:Real}`, `[s]`) time vector 
- `Δt`: (`::AbstractVector{T<:Real}`, `[s]`) delta time vector 
- `Ez2`: (`::AbstractVector{T<:Real}`, `[T/m^2]`) 2nd-order Z^2 eddy current field vector 
- `Ezx`: (`::AbstractVector{T<:Real}`, `[T/m^2]`) 2nd-order ZX eddy current field vector 
- `Ezy`: (`::AbstractVector{T<:Real}`, `[T/m^2]`) 2nd-order ZY eddy current field vector 
- `Exy`: (`::AbstractVector{T<:Real}`, `[T/m^2]`) 2nd-order XY eddy current field vector 
- `Ex2y2`: (`::AbstractVector{T<:Real}`, `[T/m^2]`) 2nd-order X^2-Y^2 eddy current field vector 

# Returns
- `seqd`: (`::DiscreteEddySequence`) DiscreteEddySequence struct 
"""

# 0.1 Type Definition
struct DiscreteEddySequence{
    T<:Real, 
    GXT<:AbstractVector{T},
    GYT<:AbstractVector{T},
    GZT<:AbstractVector{T},
    B1T<:AbstractVector{Complex{T}},
    ΔFT<:AbstractVector{T},
    ΨT<:AbstractVector{T},
    ADCT<:AbstractVector{Bool},
    EXCT<:AbstractVector{Bool},
    TT<:AbstractVector{T},
    ΔTT<:AbstractVector{T},
    # Add in eddy current params
    EZ2T<:AbstractVector{T},
    EZXT<:AbstractVector{T},
    EZYT<:AbstractVector{T},
    EXYT<:AbstractVector{T},
    EX2Y2T<:AbstractVector{T}
}

    # Standard fields (oth- and 1st-order terms included)
    Gx::GXT
    Gy::GYT
    Gz::GZT
    B1::B1T
    Δf::ΔFT
    ψ::ΨT
    ADC::ADCT
    excitation_bool::EXCT
    t::TT
    Δt::ΔTT
    # 2nd-order eddy current fields
    Ez2::EZ2T
    Ezx::EZXT
    Ezy::EZYT
    Exy::EXYT
    Ex2y2::EX2Y2T
end

# 0.2 Storage helpers
function DiscreteEddySequence(t::AbstractVector{T}=Float64[]) where {T<:Real}
    n = length(t)
    return DiscreteEddySequence(
        zeros(T, n), zeros(T, n), zeros(T, n),                  # Gx, Gy, Gz
        zeros(Complex{T}, n), zeros(T, n), zeros(T, n),         # B1, Δf, ψ
        fill(false, n), Bool[],                                 # ADC, excitation_bool
        t, similar(t, 0),                                       # t, Δt
        zeros(T, n), zeros(T, n), zeros(T, n), zeros(T, n), zeros(T, n) # 2nd-order terms
    )
end

table_columns(seqd::DiscreteEddySequence) = (
    seqd.Gx, seqd.Gy, seqd.Gz, 
    seqd.Ez2, sqed.Ezx, seqd.Ezy, seqd.Exy, seqd.Ex2y2,
    seqd.B1, seqd.Δf, seqd.ψ, seqd.ADC
)

# 0.3 Indexing and iteration
Base.length(seq::DiscreteEddySequence) = length(seq.Δt)
Base.getindex(seq::DiscreteEddySequence, i::Integer) = begin
    DiscreteEddySequence(seq.Gx[i, :],
                     seq.Gy[i, :],
                     seq.Gz[i, :],
                     seq.B1[i, :],
                     seq.Δf[i, :],
                     seq.ψ[i, :],
                     seq.ADC[i, :],
                     seq.excitation_bool[i, :],
                     seq.t[i, :],
                     seq.Δt[i, :],
                     seq.Ez2[i, :],
                     seq.Ezx[i, :],
                     seq.Ezy[i, :],
                     seq.Exy[i, :],
                     seq.Ex2y2[i, :])
end

Base.getindex(seq::DiscreteEddySequence, i::UnitRange) = begin
    intervals = i.start:i.stop-1
    DiscreteEddySequence(seq.Gx[i],
                     seq.Gy[i],
                     seq.Gz[i],
                     seq.B1[i],
                     seq.Δf[i],
                     seq.ψ[i],
                     seq.ADC[i],
                     seq.excitation_bool[intervals],
                     seq.t[i],
                     seq.Δt[intervals],
                     seq.Ez2[i],
                     seq.Ezx[i],
                     seq.Ezy[i],
                     seq.Exy[i],
                     seq.Ex2y2[i])
end
Base.view(seq::DiscreteEddySequence, i::UnitRange) = @views begin
    intervals = i.start:i.stop-1
    DiscreteEddySequence(seq.Gx[i],
                     seq.Gy[i],
                     seq.Gz[i],
                     seq.B1[i],
                     seq.Δf[i],
                     seq.ψ[i],
                     seq.ADC[i],
                     seq.excitation_bool[intervals],
                     seq.t[i],
                     seq.Δt[intervals],
                     seq.Ez2[i],
                     seq.Ezx[i],
                     seq.Ezy[i],
                     seq.Exy[i],
                     seq.Ex2y2[i])
end
Base.iterate(seq::DiscreteEddySequence) = (seq[1], 2)
Base.iterate(seq::DiscreteEddySequence, i) = (i <= length(seq)) ? (seq[i], i+1) : nothing

# 0.4 Event-state predicates
is_GR_on(seq::DiscreteEddySequence) =  any(!iszero, seq.Gx) || any(!iszero, seq.Gy) || any(!iszero, seq.Gz) ||
                                   any(!iszero, seq.Ez2) || any(!iszero, seq.Ezx) || any(!iszero, seq.Ezy) ||
                                   any(!iszero, seq.Exy) || any(!iszero, seq.Ex2y2)
is_RF_on(seq::DiscreteEddySequence) =  any(seq.excitation_bool)
is_ADC_on(seq::DiscreteEddySequence) = any(seq.ADC)
is_GR_off(seq::DiscreteEddySequence) =  !is_GR_on(seq)
is_RF_off(seq::DiscreteEddySequence) =  !is_RF_on(seq)
is_ADC_off(seq::DiscreteEddySequence) = !is_ADC_on(seq)