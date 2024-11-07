// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <limits>
#include <memory>
#include <vector>

#include "DataStructures/DataVector.hpp"
#include "Domain/CoordinateMaps/TimeDependent/Shape.hpp"
#include "Domain/CoordinateMaps/TimeDependent/ShapeMapTransitionFunctions/SphereTransition.hpp"
#include "Domain/FunctionsOfTime/FunctionOfTime.hpp"
#include "Domain/FunctionsOfTime/PiecewisePolynomial.hpp"
#include "Helpers/Domain/CoordinateMaps/TestMapHelpers.hpp"

namespace domain::CoordinateMaps::ShapeMapTransitionFunctions {

SPECTRE_TEST_CASE("Unit.Domain.CoordinateMaps.Shape.SphereTransition",
                  "[Domain][Unit]") {
  constexpr double eps = std::numeric_limits<double>::epsilon() * 100;
  const size_t l_max = 4;
  const size_t num_coefs = 2 * square(l_max + 1);
  const double time = 1.3;

  domain::FunctionsOfTimeMap functions_of_time{};
  functions_of_time["Shape"] =
      std::make_unique<domain::FunctionsOfTime::PiecewisePolynomial<0>>(
          0.0, std::array{DataVector{num_coefs, 1.e-3}}, 10.0);

  {
    INFO("Sphere transition");
    const SphereTransition sphere_transition{2., 4.};
    const domain::CoordinateMaps::TimeDependent::Shape shape_map{
        std::array{0.0, 0.0, 0.0}, l_max, l_max,
        std::make_unique<SphereTransition>(sphere_transition), "Shape"};

    const std::vector<std::array<double, 3>> test_points{
        {2., 0., 0.}, {2. - eps, 0., 0.}, {1., 0., 0.}, {3., 0., 0.},
        {4., 0., 0.}, {4. + eps, 0., 0.}, {5., 0., 0.}};
    const std::vector<double> function_values{1.0, 1.0, 1.0, 0.5,
                                              0.0, 0.0, 0.0};

    for (size_t i = 0; i < test_points.size(); i++) {
      CHECK(sphere_transition(test_points[i]) == approx(function_values[i]));
      test_inverse_map(shape_map, test_points[i], time, functions_of_time);
    }
  }
  {
    INFO("Reverse sphere transition");
    const SphereTransition sphere_transition{2., 4., true};

    const domain::CoordinateMaps::TimeDependent::Shape shape_map{
        std::array{0.0, 0.0, 0.0}, 4, 4,
        std::make_unique<SphereTransition>(sphere_transition), "Shape"};

    const std::vector<std::array<double, 3>> test_points{
        {2., 0., 0.}, {2. - eps, 0., 0.}, {1., 0., 0.}, {3., 0., 0.},
        {4., 0., 0.}, {4. + eps, 0., 0.}, {5., 0., 0.}};
    const std::vector<double> function_values{0.0, 0.0, 0.0, 0.5,
                                              1.0, 1.0, 1.0};

    for (size_t i = 0; i < test_points.size(); i++) {
      CHECK(sphere_transition(test_points[i]) == approx(function_values[i]));
      test_inverse_map(shape_map, test_points[i], time, functions_of_time);
    }
  }
}

}  // namespace domain::CoordinateMaps::ShapeMapTransitionFunctions
