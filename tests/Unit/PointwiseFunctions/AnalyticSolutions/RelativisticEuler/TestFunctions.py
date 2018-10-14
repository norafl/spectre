# Distributed under the MIT License.
# See LICENSE.txt for details.

import numpy as np


# Functions for testing FishboneMoncriefDisk.cpp
def delta(r_sqrd, m, a):
    return r_sqrd - 2.0 * m * np.sqrt(r_sqrd) + a**2


def sigma(r_sqrd, sin_theta_sqrd, a):
    return r_sqrd + (1.0 - sin_theta_sqrd) * a**2


def ucase_a(r_sqrd, sin_theta_sqrd, m, a):
    return ((r_sqrd + a**2)**2 - delta(r_sqrd, m, a) * sin_theta_sqrd * a**2)


def boyer_lindquist_gtf(r_sqrd, sin_theta_sqrd, m, a):
    return (-2.0 * m * np.sqrt(r_sqrd) * a * sin_theta_sqrd / sigma(
        r_sqrd, sin_theta_sqrd, a))


def boyer_lindquist_gff(r_sqrd, sin_theta_sqrd, m, a):
    return (ucase_a(r_sqrd, sin_theta_sqrd, m, a) * sin_theta_sqrd / sigma(
        r_sqrd, sin_theta_sqrd, a))


def boyer_lindquist_r_sqrd(x, a):
    half_diff = 0.5 * (x[0]**2 + x[1]**2 + x[2]**2 - a**2)
    return half_diff + np.sqrt(half_diff**2 + a**2 * x[2]**2)


def boyer_lindquist_sin_theta_sqrd(z_sqrd, r_sqrd):
    return 1.0 - z_sqrd / r_sqrd


def kerr_schild_h(x, m, a):
    r_sqrd = boyer_lindquist_r_sqrd(x, a)
    return m * r_sqrd * np.sqrt(r_sqrd) / (r_sqrd**2 + a**2 * x[2]**2)


def kerr_schild_spatial_null_form(x, m, a):
    r_sqrd = boyer_lindquist_r_sqrd(x, a)
    r = np.sqrt(r_sqrd)
    denom = 1.0 / (r_sqrd + a**2)
    return np.array([(r * x[0] + a * x[1]) * denom,
                     (r * x[1] - a * x[0]) * denom, x[2] / r])


def kerr_schild_lapse(x, m, a):
    null_vector_0 = -1.0
    return np.sqrt(
        1.0 /
        (1.0 + 2.0 * kerr_schild_h(x, m, a) * null_vector_0 * null_vector_0))


def kerr_schild_shift(x, m, a):
    null_vector_0 = -1.0
    return ((-2.0 * kerr_schild_h(x, m, a) * null_vector_0 * kerr_schild_lapse(
        x, m, a)**2) * kerr_schild_spatial_null_form(x, m, a))


def kerr_schild_spatial_metric(x, m, a):
    prefactor = 2.0 * kerr_schild_h(x, m, a)
    null_form = kerr_schild_spatial_null_form(x, m, a)
    return np.identity(x.size) + prefactor * np.outer(null_form, null_form)


def angular_momentum(m, a, rmax):
    return (
        np.sqrt(m) * ((np.power(rmax, 1.5) + a * np.sqrt(m)) *
                      (a**2 - 2.0 * a * np.sqrt(m) * np.sqrt(rmax) + rmax**2) /
                      (2.0 * a * np.sqrt(m) * np.power(rmax, 1.5) +
                       (rmax - 3.0 * m) * rmax**2)))


def angular_velocity(angular_momentum, r_sqrd, sin_theta_sqrd, m, a):
    prefactor = (
        2.0 * angular_momentum * delta(r_sqrd, m, a) * sin_theta_sqrd /
        np.power(boyer_lindquist_gff(r_sqrd, sin_theta_sqrd, m, a), 2.0))
    return (
        prefactor / (1.0 + np.sqrt(1.0 + 2.0 * angular_momentum * prefactor)) -
        (boyer_lindquist_gtf(r_sqrd, sin_theta_sqrd, m, a) /
         boyer_lindquist_gff(r_sqrd, sin_theta_sqrd, m, a)))


def u_t(angular_momentum, r_sqrd, sin_theta_sqrd, m, a):
    return np.sqrt(
        angular_momentum /
        (boyer_lindquist_gtf(r_sqrd, sin_theta_sqrd, m, a) +
         angular_velocity(angular_momentum, r_sqrd, sin_theta_sqrd, m, a) *
         boyer_lindquist_gff(r_sqrd, sin_theta_sqrd, m, a)))


def potential(angular_momentum, r_sqrd, sin_theta_sqrd, m, a):
    return (angular_momentum * angular_velocity(
        angular_momentum, r_sqrd, sin_theta_sqrd, m, a) - np.log(
            u_t(angular_momentum, r_sqrd, sin_theta_sqrd, m, a)))


def specific_enthalpy(x, t, black_hole_mass, black_hole_spin, r_in, r_max,
                      polytropic_constant, polytropic_exponent):
    l = angular_momentum(black_hole_mass, black_hole_spin, r_max)
    Win = potential(l, r_in * r_in, 1.0, black_hole_mass, black_hole_spin)
    r_sqrd = boyer_lindquist_r_sqrd(x, black_hole_spin)
    sin_theta_sqrd = boyer_lindquist_sin_theta_sqrd(x[2] * x[2], r_sqrd)
    result = 1.0
    if (np.sqrt(r_sqrd * sin_theta_sqrd) >= r_in):
        W = potential(l, r_sqrd, sin_theta_sqrd, black_hole_mass,
                      black_hole_spin)
        if (W < Win):
            result = np.exp(Win - W)

    return result


def rest_mass_density(x, t, black_hole_mass, black_hole_spin, r_in, r_max,
                      polytropic_constant, polytropic_exponent):
    return np.power(
        (polytropic_exponent - 1.0) *
        (specific_enthalpy(x, t, black_hole_mass, black_hole_spin, r_in, r_max,
                           polytropic_constant, polytropic_exponent) - 1.0) /
        (polytropic_exponent * polytropic_constant),
        1.0 / (polytropic_exponent - 1.0))


def specific_internal_energy(x, t, black_hole_mass, black_hole_spin, r_in,
                             r_max, polytropic_constant, polytropic_exponent):
    return (polytropic_constant * np.power(
        rest_mass_density(x, t, black_hole_mass, black_hole_spin, r_in, r_max,
                          polytropic_constant, polytropic_exponent),
        polytropic_exponent - 1.0) / (polytropic_exponent - 1.0))


def pressure(x, t, black_hole_mass, black_hole_spin, r_in, r_max,
             polytropic_constant, polytropic_exponent):
    return (polytropic_constant * np.power(
        rest_mass_density(x, t, black_hole_mass, black_hole_spin, r_in, r_max,
                          polytropic_constant, polytropic_exponent),
        polytropic_exponent))


def spatial_velocity(x, t, black_hole_mass, black_hole_spin, r_in, r_max,
                     polytropic_constant, polytropic_exponent):
    l = angular_momentum(black_hole_mass, black_hole_spin, r_max)
    Win = potential(l, r_in * r_in, 1.0, black_hole_mass, black_hole_spin)
    r_sqrd = boyer_lindquist_r_sqrd(x, black_hole_spin)
    sin_theta_sqrd = boyer_lindquist_sin_theta_sqrd(x[2] * x[2], r_sqrd)

    result = np.array([0.0, 0.0, 0.0])
    if (np.sqrt(r_sqrd * sin_theta_sqrd) >= r_in):
        W = potential(l, r_sqrd, sin_theta_sqrd, black_hole_mass,
                      black_hole_spin)
        if (W < Win):
            result += ((np.array([-x[1], x[0], 0.0]) * angular_velocity(
                l, r_sqrd, sin_theta_sqrd, black_hole_mass, black_hole_spin) +
                        kerr_schild_shift(x, black_hole_mass, black_hole_spin))
                       / kerr_schild_lapse(x, black_hole_mass,
                                           black_hole_spin))
    return result


# End functions for testing FishboneMoncriefDisk.cpp
