export BlochEddy

"""
Simulation method for sequences with spatial eddy currents.
"""
struct BlochEddy <: SimulationMethod end

"""
    run_spin_precession!(obj, seq, sig, M, sim_method, groupsize, backend, prealloc)

Executes spin precession using DiscreteEddySequence struct, evaluating 2-nd order
spatial harmonics while benefiting from baked 0th- and 1st-orer gradients.
"""
function run_spin_precession!(
    p::Phantom{T},
    seq::DiscreteEddySequence{T},
    sig::AbstractArray{Complex{T}},
    M::Mag{T},
    sys,
    sim_method::BlochEddy,
    groupsize,
    backend::KA.CPU,
    prealloc::PreallocResult{T}
) where {T<:Real}

    # Rename arrays
    Bz_old = prealloc.Bz_old
    Bz_new = prealloc.Bz_new
    ϕ = prealloc.ϕ
    Mxy = prealloc.M.xy
    ΔBz = prealloc.ΔBz
    
    # Initialize
    fill!(ϕ, zero(T))
    block_time = zero(T)
    sample = 1
    x, y, z = spin_coordinates!(
        prealloc.coordinates, p.motion, p.x, p.y, p.z, seq.t[1],
    )

    # Effectivfe field (start of block)
    @. Bz_old = (x * seq.Gx[1] + y * seq.Gy[1] + z * seq.Gz[1] + ΔBz +
                (x * y * seq.Exy[1]) + ((z^2 - 0.5 * (x^2 + y^2)) * seq.Ez2[1]) +
                (z * x * seq.Ezx[1]) + (z * y * seq.Ezy[1]) + ((x^2 - y^2) * seq.Ex2y2[1]))
    
    # Simulation
    for i in eachindex(seq.Δt)
        # Motion
        x, y, z = spin_coordinates!(
            prealloc.coordinates, p.motion, p.x, p.y, p.z, seq.t[1],
        )

        # Effective field (time step)
        @. Bz_new = (x * seq.Gx[i + 1] + y * seq.Gy[i + 1] + z * seq.Gz[i + 1] + ΔBz +
                    (x * y * seq.Exy[i + 1]) + ((z^2 - 0.5 * (x^2 + y^2)) * seq.Ez2[i + 1]) +
                    (z * x * seq.Ezx[i + 1]) + (z * y * seq.Ezy[i + 1]) + ((x^2 - y^2) * seq.Ex2y2[i + 1]))

        # Rotation
        @. ϕ += (Bz_old + Bz_new) * T(-π * γ) * seq.Δt[i]
        block_time += seq.Δt[i]

        # Acquired signal
        if seq.ADC[i + 1]
            # Update signal
            @. Mxy = exp(-block_time / p.T2) * M.xy * cis(ϕ)
            # Reset Spin-State (Magnetization). Only for FlowPath
            outflow_spin_reset!(Mxy, seq.t[i + 1], p.motion)
            update_sensitivities!(prealloc.sens, sys.receiver, (x, y, z), p.motion)
            acquire_signal!(
                @view(sig[sample, :]), Mxy, prealloc.sens, (x, y, z),
            )
            sample += 1
        end
        Bz_old .= Bz_new
    end

    # Final spin-state
    @. M.xy = M.xy * exp(-block_time / p.T2) * cis(ϕ)
    @. M.z = M.z * exp(-block_time / p.T1) + p.ρ * (T(1) - exp(-block_time / p.T1))
    outflow_spin_reset!(M,  seq.t', p.motion; replace_by=p.ρ)
    return nothing
end

"""
    run_spin_excitation!(obj, seq, sig, M, sim_method, groupsize, backend, prealloc)
***********************
"""
function run_spin_excitation!(
    p::Phantom{T},
    seq::DiscreteEddySequence{T},
    sig::AbstractArray{Complex{T}},
    M::Mag{T},
    sys,
    sim_method::BlochEddy,
    groupsize,
    backend::KA.CPU,
    prealloc::BlochCPUPrealloc
) where {T<:Real}

    # Rename arrays
    Bz = prealloc.Bz_old
    B = prealloc.Bz_new
    φ_half = prealloc.ϕ
    α = prealloc.Rot.α
    β = prealloc.Rot.β
    ΔBz = prealloc.ΔBz
    Maux_xy = prealloc.M.xy
    Maux_z = prealloc.M.z

    # Initialise
    ψ_start = seq.ψ[1]
    if !iszero(ψ_start)
        @. M.xy = M.xy * cis(-ψ_start)
    end

    # Simulation
    for i in eachindex(seq.Δt)
        # Motion
        x, y, z = spin_coordinates!(
            prealloc.coordinates, p.motion, p.x, p.y, p.z, seq.t[1],
        )

        # Effective field
        @. Bz = ((seq.Gx[i] * x + seq.Gy[i] * y + seq.Gz[i] * z) + 
                (x * y * seq.Exy[i]) + ((z^2 - 0.5 * (x^2 + y^2)) * seq.Ez2[i]) +
                (z * x * seq.Ezx[i]) + (z * y * seq.Ezy[i]) + ((x^2 - y^2) * seq.Ex2y2[i]) +
                ΔBz - seq.Δf[i] / T(γ)) # ΔB_0 = (B_0 - ω_rf/γ), Need to add a component here to model scanner's dB0(x,y,z)
        @. B = sqrt(abs2(seq.B1[i]) + Bz^2)

        # Spinor Rotation
        @. φ_half = T(-π * γ) * (B * seq.Δt[i]) # TODO: Use trapezoidal integration here (?),  this is just Forward Euler
        @. α = cos(φ_half)
        @. B = sin(φ_half) / (B + (B == 0) * eps(T))
        @. α -= Complex{T}(im) * Bz * B
        @. β = -Complex{T}(im) * seq.B1[i] * B
        mul!(Spinor(α, β), M, Maux_xy, Maux_z)

        # Relaxation
        @. M.xy = M.xy * exp(-seq.Δt[i] / p.T2)
        @. M.z = M.z * exp(-seq.Δt[i] / p.T1) + p.ρ * (T(1) - exp(-seq.Δt[i] / p.T1))
        # Reset Spin-State (Magnetization). Only for FlowPath
        outflow_spin_reset_at!(M, seq.t, i + 1, p.motion; replace_by=p.ρ)

        # Acquire signal
        if seq.ADC[i + 1] # ADC at the end of the time step
            coords = spin_coordinates!(
                prealloc.coordinates, p.motion, p.x, p.y, p.z, seq.t[i + 1],
            )
            update_sensitivities!(prealloc.sens, sys.receiver, coords, p.motion)
            acquire_signal!(@view(sig[sample, :]), M.xy, prealloc.sens, coords)
            sample += 1
        end
    end

    # RF frame -> Rotating frame
    ψ_end = seq.ψ[end]
    if !iszero(ψ_end)
        @. M.xy = M.xy * cis(ψ_end)
    end

    return nothing
end