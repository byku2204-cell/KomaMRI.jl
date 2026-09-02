"""
    seqd_new = calculate_eddy(seqd, τ, coefficients)

Calculates and applies 0th, 1st, and 2nd-order eddy currents to a discrete sequence.
The 0th- and 1st-order terms are baked directly into the primary gradients and frequency offsets
to optimise memory and processing speed in the Bloch solver.

The resulting hardware imperfection can be used to simulate spatial distortions and
global phase shifts, amongst other eddy current effects.

# Arguments
- `seqd`: (`::DiscreteSequence`) the input discrete sequence containing ideal waveforms.
- `τ`: (`::Real`, `[s]`) the exponential decay time constant.
- `coefficients`: (`::AbstractMatrix{<:Real}`) 9x3 hardware coupling coefficients.
    Columns represent the driving gradient (X, Y, Z).
    Rows 1-4 represent the resulting 0th- and 1st-order fields (B0, X, Y, Z).
    Rows 5-9 represent the resulting 2nd-order fields (Z^2, ZX, ZY, XY, X^2-Y^2).

# Returns
- `seqd_eddy`: (`::DiscreteEddySequence`) a new sequence object containing the modified waveforms.
    0th- and 1st-order terms injected directly into `Gx`, `Gy`, `Gz`, and `Δf` arrays.
    2nd-order terms are injected into the `Ez2`, `Ezx`, `Ezy`, `Exy`, and `Ex2y2` arrays.
"""

function calculate_eddy(seqd, τ, coefficients)
    # Extract arrays
    Gx, Gy, Gz, t = seqd.Gx, seqd.Gy, seqd.Gz, seqd.t
    N = length(Gx)

    # Return empty eddy seq if empty seqd inputted
    if isempty(t)
        return DiscreteEddySequence(
            Gx, Gy, Gz, seqd.B1, seqd.Δf, seqd.ψ, seqd.ADC, seqd.excitation_bool, t, seqd.Δt, 
            zeros(eltype(Gx), 0), zeros(eltypr(Gx), 0), zeros(eltypr(Gx), 0), zeros(eltypr(Gx), 0), zeros(eltypr(Gx), 0)
        )
    end

    # Calculate amplitude steps
    dGx = [0.0; diff(Gx)]
    dGy = [0.0; diff(Gy)]
    dGz = [0.0, diff(Gz)]

    C = coefficients

    # Empty arrays for 0th and 1st-order tracking
    Eb0 = zeros(eltype(Gx), N)
    Ex = zeros(eltype(Gx), N)
    Ey = zeros(eltype(Gx), N)
    Ez = zeros(eltype(Gx), N)

    # Empty arrays for 2nd-order terms
    Ez2 = zeros(eltype(Gx), N)
    Ezx = zeros(eltype(Gx), N)
    Ezy = zeros(eltype(Gx), N)
    Exy = zeros(eltype(Gx), N)
    Ex2y2 = zeros(eltype(Gx), N)

    # Recursive integration
    for i in 2:N
        # Elapsed time between steps
        dt_elapsed = t[i] - t[i-1]

        # Decay between steps
        decay = exp(-dt_elapsed / τ)

        # 0th- and 1st-order terms (Rows 1-4)
        Eb0[i] = Eb0[i-1]*decay + (dGx[i]*C[1, 1] + dGy[i]*C[1, 2] + dGz[i]*C[1, 3])
        Ex[i] = Ex[i-1]*decay + (dGx[i]*C[2, 1] + dGy[i]*C[2, 2] + dGz[i]*C[2, 3])
        Ey[i] = Ey[i-1]*decay + (dGx[i]*C[3, 1] + dGy[i]*C[3, 2] + dGz[i]*C[3, 3])
        Ez[i] = Ez[i-1]*decay + (dGx[i]*C[4, 1] + dGy[i]*C[4, 2] + dGz[i]*C[4, 3])

        # 2nd-order terms (Rows 5-9)
        Ez2[i] = Ez2[i-1]*decay + (dGx[i]*C[5, 1] + dGy[i]*C[5, 2] + dGz[i]*C[5, 3])
        Ezx[i] = Ezx[i-1]*decay + (dGx[i]*C[6, 1] + dGy[i]*C[6, 2] + dGz[i]*C[6, 3])
        Ezy[i] = Ezy[i-1]*decay + (dGx[i]*C[7, 1] + dGy[i]*C[7, 2] + dGz[i]*C[7, 3])
        Exy[i] = Exy[i-1]*decay + (dGx[i]*C[8, 1] + dGy[i]*C[8, 2] + dGz[i]*C[8, 3])
        Ex2y2[i] = Ex2y2[i-1]*decay + (dGx[i]*C[9, 1] + dGy[i]*C[9, 2] + dGz[i]*C[9, 3])
    end

    # Merge 1st-order error tails into ideal spatial gradients
    Gx_final = Gx .+ Ex
    Gy_final = Gy .+ Ey
    Gz_final = Gz .+ Ez 

    # Merge 0th-order error tail into frequency array (converting T to Hz)
    Δf_final = seqd.Δf .- (Eb0 .* γ)

    return DiscreteEddySequence(
        Gx_final, Gy_final, Gz_final, 
        seqd.B1, Δf_final, seqd.ψ, seqd.ADC, seqd.excitation_bool, t, seqd.Δt,
        Ez2, Ezx, Ezy, Exy, Ex2y2
    )
end