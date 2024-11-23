// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <random>

#include "Evolution/Systems/CurvedScalarWave/Worldtube/SelfForce.hpp"
#include "Framework/CheckWithRandomValues.hpp"
#include "Framework/SetupLocalPythonEnvironment.hpp"
#include "Framework/TestCreation.hpp"
#include "Framework/TestHelpers.hpp"
#include "Utilities/TMPL.hpp"

namespace CurvedScalarWave::Worldtube {
namespace {

void test_self_force_acceleration() {
  pypp::check_with_random_values<1>(
      static_cast<tnsr::I<double, 3> (*)(
          const Scalar<double>&, const tnsr::i<double, 3>&,
          const tnsr::I<double, 3>&, const double, const double,
          const tnsr::AA<double, 3>&, const Scalar<double>&)>(
          self_force_acceleration<3>),
      "SelfForce", "self_force_acceleration", {{{-2.0, 2.0}}}, 1);
}

void test_self_force_per_mass() {
  pypp::check_with_random_values<1>(
      static_cast<tnsr::A<double, 3> (*)(
          const tnsr::a<double, 3>&, const tnsr::A<double, 3>&, const double,
          const double, const tnsr::AA<double, 3>&)>(self_force_per_mass<3>),
      "SelfForce", "self_force_per_mass", {{{-2.0, 2.0}}}, 1);
}

void test_dt_self_force_per_mass() {
  pypp::check_with_random_values<1>(
      static_cast<tnsr::A<double, 3> (*)(
          const tnsr::a<double, 3>&, const tnsr::a<double, 3>&,
          const tnsr::A<double, 3>&, const tnsr::A<double, 3>&, const double,
          const double, const tnsr::AA<double, 3>&,
          const tnsr::AA<double, 3>&)>(dt_self_force_per_mass<3>),
      "SelfForce", "dt_self_force_per_mass", {{{-2.0, 2.0}}}, 1);
}

void test_dt2_self_force_per_mass() {
  pypp::check_with_random_values<1>(
      static_cast<tnsr::A<double, 3> (*)(
          const tnsr::a<double, 3>&, const tnsr::a<double, 3>&,
          const tnsr::a<double, 3>&, const tnsr::A<double, 3>&,
          const tnsr::A<double, 3>&, const tnsr::A<double, 3>&, const double,
          const double, const tnsr::AA<double, 3>&, const tnsr::AA<double, 3>&,
          const tnsr::AA<double, 3>&)>(dt2_self_force_per_mass<3>),
      "SelfForce", "dt2_self_force_per_mass", {{{-2.0, 2.0}}}, 1);
}

void test_Du_self_force_per_mass() {
  pypp::check_with_random_values<1>(
      static_cast<tnsr::A<double, 3> (*)(
          const tnsr::A<double, 3>&, const tnsr::A<double, 3>&,
          const tnsr::A<double, 3>&, const tnsr::Abb<double, 3>&)>(
          Du_self_force_per_mass<3>),
      "SelfForce", "Du_self_force_per_mass", {{{-2.0, 2.0}}}, 1);
}

void test_dt_Du_self_force_per_mass() {
  pypp::check_with_random_values<1>(
      static_cast<tnsr::A<double, 3> (*)(
          const tnsr::A<double, 3>&, const tnsr::A<double, 3>&,
          const tnsr::A<double, 3>&, const tnsr::A<double, 3>&,
          const tnsr::A<double, 3>&, const tnsr::Abb<double, 3>&,
          const tnsr::Abb<double, 3>&)>(dt_Du_self_force_per_mass<3>),
      "SelfForce", "dt_Du_self_force_per_mass", {{{-2.0, 2.0}}}, 1);
}

void test_turn_on_function() {
  pypp::check_with_random_values<1>(
      static_cast<double (*)(const double time,
                             const double turn_on_timescale)>(turn_on_function),
      "SelfForce", "turn_on_function", {{{0., 2.0}}}, 1);
}

void test_turn_on_function_derivatives() {
  MAKE_GENERATOR(generator);
  std::uniform_real_distribution dist(0.001, 10000.);
  const double timescale = dist(generator);
  const double time = dist(generator);
  const std::array<double, 1> time_array{{time}};

  const auto func_helper = [&timescale](const std::array<double, 1>& time_in) {
    return turn_on_function(time_in[0], timescale);
  };
  const auto dt_func_helper =
      [&timescale](const std::array<double, 1>& time_in) {
        return dt_turn_on_function(time_in[0], timescale);
      };

  const double dt = 1e-5;
  const Approx custom_approx = Approx::custom().epsilon(1e-10).scale(1.0);
  const auto numerical_deriv_func =
      numerical_derivative(func_helper, time_array, 0, dt);
  CHECK(dt_turn_on_function(time, timescale) ==
        custom_approx(numerical_deriv_func));

  const auto numerical_deriv_dt_func =
      numerical_derivative(dt_func_helper, time_array, 0, dt);
  CHECK(dt2_turn_on_function(time, timescale) ==
        custom_approx(numerical_deriv_dt_func));
}

SPECTRE_TEST_CASE("Unit.Evolution.Systems.CSW.Worldtube.SelfForce",
                  "[Unit][Evolution]") {
  pypp::SetupLocalPythonEnvironment local_python_env{
      "Evolution/Systems/CurvedScalarWave/Worldtube"};
  test_self_force_acceleration();
  test_self_force_per_mass();
  test_dt_self_force_per_mass();
  test_dt2_self_force_per_mass();
  test_Du_self_force_per_mass();
  test_dt_Du_self_force_per_mass();
  test_turn_on_function();
  test_turn_on_function_derivatives();
}
}  // namespace
}  // namespace CurvedScalarWave::Worldtube
