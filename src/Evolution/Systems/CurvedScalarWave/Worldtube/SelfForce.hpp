// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/Tensor/Tensor.hpp"
#include "Utilities/Gsl.hpp"

namespace CurvedScalarWave::Worldtube {

namespace detail {
void check_positive_time_and_timescale(double t_minus_turn_on,
                                       double turn_on_timescale);

}  // namespace detail

/// @{
/*!
 * \brief Computes the coordinate acceleration due to the scalar self-force onto
 * the charge.
 *
 * \details It is given by
 *
 * \begin{equation}
 * (u^0)^2 \ddot{x}^i_p  = \frac{q}{\mu}(g^{i
 * \alpha} - \dot{x}^i_p g^{0 \alpha} ) \partial_\alpha \Psi^R
 * \end{equation}
 *
 * where $\dot{x}^i_p$ is the position of the scalar charge, $\Psi^R$ is the
 * regular field, $q$ is the particle's charge, $\mu$ is the particle's mass,
 * $u^\alpha$ is the four-velocity and $g^{\alpha \beta}$ is the inverse
 * spacetime metric in the inertial frame, evaluated at the position of the
 * particle. An overdot denotes a derivative with respect to coordinate time.
 * Greek indices are spacetime indices and Latin indices are purely spatial.
 * Note that the coordinate geodesic acceleration is NOT included.
 */
template <size_t Dim>
void self_force_acceleration(
    gsl::not_null<tnsr::I<double, Dim>*> self_force_acc,
    const Scalar<double>& dt_psi_monopole,
    const tnsr::i<double, Dim>& psi_dipole,
    const tnsr::I<double, Dim>& particle_velocity, double particle_charge,
    double particle_mass, const tnsr::AA<double, Dim>& inverse_metric,
    const Scalar<double>& dilation_factor);

template <size_t Dim>
tnsr::I<double, Dim> self_force_acceleration(
    const Scalar<double>& dt_psi_monopole,
    const tnsr::i<double, Dim>& psi_dipole,
    const tnsr::I<double, Dim>& particle_velocity, double particle_charge,
    double particle_mass, const tnsr::AA<double, Dim>& inverse_metric,
    const Scalar<double>& dilation_factor);

/// @}
/*!
 * \brief Computes the scalar self-force per unit mass
 *
 * \details It is given by
 * \begin{equation}
 * f^\alpha = \frac{q}{\mu} (g^{\alpha \beta} + u^\alpha u^\beta) \partial_\beta
 * \Psi^R
 * \end{equation}
 * where $\Psi^R$ is the regular field at the position of the particle, $q$ is
 * the particle's charge, $\mu$ is the particle's mass, $u^\alpha$ is the
 * four-velocity and $g^{\alpha \beta}$ is the inverse spacetime metric in the
 * inertial frame, evaluated at the position of the particle.
 */
template <size_t Dim>
tnsr::A<double, Dim> self_force_per_mass(
    const tnsr::a<double, Dim>& d_psi,
    const tnsr::A<double, Dim>& four_velocity, double particle_charge,
    double particle_mass, const tnsr::AA<double, Dim>& inverse_metric);

/*!
 * \brief Computes the first time derivative of scalar self-force per unit mass,
 * see `self_force_per_mass`, by applying the chain rule.
 */
template <size_t Dim>
tnsr::A<double, Dim> dt_self_force_per_mass(
    const tnsr::a<double, Dim>& d_psi, const tnsr::a<double, Dim>& dt_d_psi,
    const tnsr::A<double, Dim>& four_velocity,
    const tnsr::A<double, Dim>& dt_four_velocity, double particle_charge,
    double particle_mass, const tnsr::AA<double, Dim>& inverse_metric,
    const tnsr::AA<double, Dim>& dt_inverse_metric);

/*!
 * \brief Computes the second time derivative of scalar self-force per unit
 * mass, see `self_force_per_mass`, by applying the chain rule.
 */
template <size_t Dim>
tnsr::A<double, Dim> dt2_self_force_per_mass(
    const tnsr::a<double, Dim>& d_psi, const tnsr::a<double, Dim>& dt_d_psi,
    const tnsr::a<double, Dim>& dt2_d_psi,
    const tnsr::A<double, Dim>& four_velocity,
    const tnsr::A<double, Dim>& dt_four_velocity,
    const tnsr::A<double, Dim>& dt2_four_velocity, double particle_charge,
    double particle_mass, const tnsr::AA<double, Dim>& inverse_metric,
    const tnsr::AA<double, Dim>& dt_inverse_metric,
    const tnsr::AA<double, Dim>& dt2_inverse_metric);

/*!
 * \brief Computes the covariant derivative of the scalar self-force per unit
 * mass $f^\alpha$, see `self_force_per_mass`, along the four velocity
 * $u^\beta$, i.e. $u^\beta \nabla_\beta f^\alpha$.
 */
template <size_t Dim>
tnsr::A<double, Dim> Du_self_force_per_mass(
    const tnsr::A<double, Dim>& self_force,
    const tnsr::A<double, Dim>& dt_self_force,
    const tnsr::A<double, Dim>& four_velocity,
    const tnsr::Abb<double, Dim>& christoffel);
/*!
 * \brief Computes the time derivative of the covariant derivative of the scalar
 * self-force per unit mass $f^\alpha$, see `Du_self_force_per_mass`, along the
 * four velocity $u^\beta$, i.e.
 * $\frac{d}{dt}u^\beta \nabla_\beta f^\alpha$.
 */
template <size_t Dim>
tnsr::A<double, Dim> dt_Du_self_force_per_mass(
    const tnsr::A<double, Dim>& self_force,
    const tnsr::A<double, Dim>& dt_self_force,
    const tnsr::A<double, Dim>& dt2_self_force,
    const tnsr::A<double, Dim>& four_velocity,
    const tnsr::A<double, Dim>& dt_four_velocity,
    const tnsr::Abb<double, Dim>& christoffel,
    const tnsr::Abb<double, Dim>& dt_christoffel);

/*!
 * \brief A function used to roll-on the self-force continuously from 0 to 1
 *
 * \details It is given by Eq.(60) of \cite Wittek:2024gxn
 * \begin{equation}
 * w(t) = 1 - \exp \left( - \left( (t - t_{\mathrm{turn_on}}) / \sigma \right)^4
 * \right), \end{equation} where $t$ is the current simulation time,
 * $t_{\mathrm{turn_on}}$ is the time where the self-force is turned on and
 * $\sigma$ dictates the timescale over which it is turned on. The function is
 * $\mathcal{C}^3$, i.e. three times continuously differentiable, assuming $w(t)
 * = 0$ for $t < t_{\mathrm{turn_on}}$.
 */
double turn_on_function(double t_minus_turn_on, double turn_on_timescale);

/*!
 * \brief The first derivative of `turn_on_function`
 */
double dt_turn_on_function(double t_minus_turn_on, double turn_on_timescale);

/*!
 * \brief The second derivative of `turn_on_function`
 */
double dt2_turn_on_function(double t_minus_turn_on, double turn_on_timescale);

}  // namespace CurvedScalarWave::Worldtube
