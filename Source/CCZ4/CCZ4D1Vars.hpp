/* GRTeclyn
 * Copyright 2022 The GRTL collaboration.
 * Please refer to LICENSE in GRTeclyn's root directory.
 */

#ifndef CCZ4D1VARS_HPP_
#define CCZ4D1VARS_HPP_

#include "FourthOrderDerivatives.hpp"
#include "StateVariables.hpp"
#include "Tensor.hpp"
#include "AMReX_Array4.H"

class CCZ4D1Vars
{
  public:
    AMREX_GPU_DEVICE CCZ4D1Vars(int ix, int iy, int iz,
                                const amrex::Array4<const amrex::Real> &state,
                                const FourthOrderDerivatives &a_deriv)
    {
        m_d1_state = a_deriv.diff1_state(ix, iy, iz, state);
    }

    amrex::GpuArray<Tensor<1, amrex::Real>, NUM_VARS> m_d1_state;

    [[nodiscard]]
    AMREX_GPU_DEVICE AMREX_FORCE_INLINE const Tensor<1, amrex::Real> &
    chi() const
    {
        return m_d1_state[c_chi];
    }

    [[nodiscard]]
    AMREX_GPU_DEVICE AMREX_FORCE_INLINE const Tensor<1, amrex::Real> &
    h(int i, int j) const
    {
        return m_d1_state[var_idx(c_h11, i, j)];
    }

    [[nodiscard]]
    AMREX_GPU_DEVICE AMREX_FORCE_INLINE const Tensor<1, amrex::Real> &K() const
    {
        return m_d1_state[c_K];
    }

    [[nodiscard]]
    AMREX_GPU_DEVICE AMREX_FORCE_INLINE const Tensor<1, amrex::Real> &
    A(int i, int j) const
    {
        return m_d1_state[var_idx(c_A11, i, j)];
    }

    [[nodiscard]]
    AMREX_GPU_DEVICE AMREX_FORCE_INLINE const Tensor<1, amrex::Real> &
    Theta() const
    {
        return m_d1_state[c_Theta];
    }

    [[nodiscard]]
    AMREX_GPU_DEVICE AMREX_FORCE_INLINE const Tensor<1, amrex::Real> &
    Gamma(int i) const
    {
        return m_d1_state[c_Gamma1 + i];
    }

    [[nodiscard]]
    AMREX_GPU_DEVICE AMREX_FORCE_INLINE const Tensor<1, amrex::Real> &
    lapse() const
    {
        return m_d1_state[c_lapse];
    }

    [[nodiscard]]
    AMREX_GPU_DEVICE AMREX_FORCE_INLINE const Tensor<1, amrex::Real> &
    shift(int i) const
    {
        return m_d1_state[c_shift1 + i];
    }

    [[nodiscard]]
    AMREX_GPU_DEVICE AMREX_FORCE_INLINE const Tensor<1, amrex::Real> &
    B(int i) const
    {
        return m_d1_state[c_B1 + i];
    }
};

#endif /* CCZ4D1VARS_HPP */