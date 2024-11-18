// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Domain/Creators/TimeDependentOptions/Sphere.hpp"

#include <array>
#include <cmath>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <variant>

#include "DataStructures/DataVector.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.tpp"
#include "Domain/CoordinateMaps/TimeDependent/ShapeMapTransitionFunctions/ShapeMapTransitionFunction.hpp"
#include "Domain/CoordinateMaps/TimeDependent/ShapeMapTransitionFunctions/SphereTransition.hpp"
#include "Domain/CoordinateMaps/TimeDependent/ShapeMapTransitionFunctions/Wedge.hpp"
#include "Domain/Creators/TimeDependentOptions/ShapeMap.hpp"
#include "Domain/FunctionsOfTime/FixedSpeedCubic.hpp"
#include "Domain/FunctionsOfTime/FunctionOfTime.hpp"
#include "Domain/FunctionsOfTime/PiecewisePolynomial.hpp"
#include "Domain/FunctionsOfTime/QuaternionFunctionOfTime.hpp"
#include "Domain/FunctionsOfTime/SettleToConstant.hpp"
#include "Domain/FunctionsOfTime/SettleToConstantQuaternion.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/Spherepack.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/KerrHorizon.hpp"
#include "Utilities/ErrorHandling/Error.hpp"

namespace domain::creators::sphere {

TimeDependentMapOptions::RotationMapOptions::RotationMapOptions() = default;
TimeDependentMapOptions::RotationMapOptions::RotationMapOptions(
    const std::array<std::array<double, 4>, 3> initial_values_in,
    const double decay_timescale_in)
    : initial_values(initial_values_in), decay_timescale(decay_timescale_in) {}

TimeDependentMapOptions::ExpansionMapOptions::ExpansionMapOptions() = default;
TimeDependentMapOptions::ExpansionMapOptions::ExpansionMapOptions(
    const std::array<double, 3> initial_values_in,
    const double decay_timescale_in,
    const std::array<double, 3> initial_values_outer_boundary_in,
    const double decay_timescale_outer_boundary_in)
    : initial_values(initial_values_in),
      decay_timescale(decay_timescale_in),
      initial_values_outer_boundary(initial_values_outer_boundary_in),
      decay_timescale_outer_boundary(decay_timescale_outer_boundary_in) {}

TimeDependentMapOptions::TranslationMapOptions::TranslationMapOptions() =
    default;
TimeDependentMapOptions::TranslationMapOptions::TranslationMapOptions(
    const std::array<std::array<double, 3>, 3> initial_values_in)
    : initial_values(initial_values_in) {}

TimeDependentMapOptions::TimeDependentMapOptions(
    const double initial_time, std::optional<ShapeMapOptions> shape_map_options,
    std::optional<RotationMapOptions> rotation_map_options,
    std::optional<ExpansionMapOptions> expansion_map_options,
    std::optional<TranslationMapOptions> translation_map_options,
    const bool transition_rot_scale_trans)
    : initial_time_(initial_time),
      shape_map_options_(std::move(shape_map_options)),
      rotation_map_options_(std::move(rotation_map_options)),
      expansion_map_options_(std::move(expansion_map_options)),
      translation_map_options_(std::move(translation_map_options)),
      transition_rot_scale_trans_(transition_rot_scale_trans) {}

std::unordered_map<std::string,
                   std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>
TimeDependentMapOptions::create_functions_of_time(
    const std::unordered_map<std::string, double>& initial_expiration_times)
    const {
  std::unordered_map<std::string,
                     std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>
      result{};

  // Get existing function of time names that are used for the maps and assign
  // their initial expiration time to infinity (i.e. not expiring)
  std::unordered_map<std::string, double> expiration_times{
      {size_name, std::numeric_limits<double>::infinity()},
      {shape_name, std::numeric_limits<double>::infinity()},
      {translation_name, std::numeric_limits<double>::infinity()}};

  // If we have control systems, overwrite these expiration times with the ones
  // supplied by the control system
  for (const auto& [name, expr_time] : initial_expiration_times) {
    expiration_times[name] = expr_time;
  }

  if (shape_map_options_.has_value()) {
    auto [shape_funcs, size_funcs] =
        time_dependent_options::initial_shape_and_size_funcs(
            shape_map_options_.value(), deformed_radius_);

    // ShapeMap FunctionOfTime
    result[shape_name] =
        std::make_unique<FunctionsOfTime::PiecewisePolynomial<2>>(
            initial_time_, std::move(shape_funcs),
            expiration_times.at(shape_name));

    // Size FunctionOfTime (used in ShapeMap)
    result[size_name] =
        std::make_unique<FunctionsOfTime::PiecewisePolynomial<3>>(
            initial_time_, std::move(size_funcs),
            expiration_times.at(size_name));
  }

  // ExpansionMap FunctionOfTime
  if (expansion_map_options_.has_value()) {
    result[expansion_name] =
        std::make_unique<FunctionsOfTime::SettleToConstant>(
            std::array<DataVector, 3>{
                {{gsl::at(expansion_map_options_.value().initial_values, 0)},
                 {gsl::at(expansion_map_options_.value().initial_values, 1)},
                 {gsl::at(expansion_map_options_.value().initial_values, 2)}}},
            initial_time_, expansion_map_options_.value().decay_timescale);

    // ExpansionMap in the Outer regionFunctionOfTime
    result[expansion_outer_boundary_name] = std::make_unique<
        FunctionsOfTime::SettleToConstant>(
        std::array<DataVector, 3>{
            {{gsl::at(
                 expansion_map_options_.value().initial_values_outer_boundary,
                 0)},
             {gsl::at(
                 expansion_map_options_.value().initial_values_outer_boundary,
                 1)},
             {gsl::at(
                 expansion_map_options_.value().initial_values_outer_boundary,
                 2)}}},
        initial_time_,
        expansion_map_options_.value().decay_timescale_outer_boundary);
  }

  DataVector initial_quaternion_value{4, 0.0};
  DataVector initial_quaternion_first_derivative_value{4, 0.0};
  DataVector initial_quaternion_second_derivative_value{4, 0.0};

  // RotationMap FunctionOfTime
  if (rotation_map_options_.has_value()) {
    for (size_t i = 0; i < 4; i++) {
      initial_quaternion_value[i] =
          gsl::at(gsl::at(rotation_map_options_.value().initial_values, 0), i);
      initial_quaternion_first_derivative_value[i] =
          gsl::at(gsl::at(rotation_map_options_.value().initial_values, 1), i);
      initial_quaternion_second_derivative_value[i] =
          gsl::at(gsl::at(rotation_map_options_.value().initial_values, 2), i);
    }
    result[rotation_name] =
        std::make_unique<FunctionsOfTime::SettleToConstantQuaternion>(
            std::array<DataVector, 3>{
                std::move(initial_quaternion_value),
                std::move(initial_quaternion_first_derivative_value),
                std::move(initial_quaternion_second_derivative_value)},
            initial_time_, rotation_map_options_.value().decay_timescale);
  }

  DataVector initial_translation_center{3, 0.0};
  DataVector initial_translation_velocity{3, 0.0};
  DataVector initial_translation_acceleration{3, 0.0};

  // Translation FunctionOfTime
  if (translation_map_options_.has_value()) {
    for (size_t i = 0; i < 3; i++) {
      initial_translation_center[i] = gsl::at(
          gsl::at(translation_map_options_.value().initial_values, 0), i);
      initial_translation_velocity[i] = gsl::at(
          gsl::at(translation_map_options_.value().initial_values, 1), i);
      initial_translation_acceleration[i] = gsl::at(
          gsl::at(translation_map_options_.value().initial_values, 2), i);
    }
    result[translation_name] =
        std::make_unique<FunctionsOfTime::PiecewisePolynomial<2>>(
            initial_time_,
            std::array<DataVector, 3>{
                {std::move(initial_translation_center),
                 std::move(initial_translation_velocity),
                 std::move(initial_translation_acceleration)}},
            expiration_times.at(translation_name));
  }

  return result;
}

void TimeDependentMapOptions::build_maps(
    const std::array<double, 3>& center, const bool filled,
    const double inner_radius, const std::vector<double>& radial_partitions,
    const double outer_radius) {
  filled_ = filled;
  if (shape_map_options_.has_value()) {
    std::unique_ptr<domain::CoordinateMaps::ShapeMapTransitionFunctions::
                        ShapeMapTransitionFunction>
        transition_func;
    if (filled_) {
      if (transition_rot_scale_trans_ and radial_partitions.size() < 2) {
        ERROR(
            "Currently at least two radial partitions are required to "
            "transition the RotScaleTrans map to zero in the outer shell "
            "when a shape map is present and the interior is filled.");
      }
      using WedgeTransition =
          domain::CoordinateMaps::ShapeMapTransitionFunctions::Wedge;
      // Shape map transitions from 0 to 1 from the inner cube to this surface
      deformed_radius_ =
          radial_partitions.empty() ? outer_radius : radial_partitions.front();
      // Shape map transitions from 1 to 0 from the deformed surface to the next
      // radial partition or to the outer radius
      const bool has_shape_rolloff = not radial_partitions.empty();
      const double shape_outer_radius =
          radial_partitions.size() > 1 ? radial_partitions[1] : outer_radius;
      // These must match the order of orientations_for_sphere_wrappings() in
      // DomainHelpers.hpp. The values must match that of Wedge::Axis
      const std::array<int, 6> axes{3, -3, 2, -2, 1, -1};
      for (size_t j = 0; j < (has_shape_rolloff ? 12 : 6); ++j) {
        if (j < 6) {
          // Reverse the transition function so the shape map goes to zero at
          // the inner cube
          transition_func = std::make_unique<WedgeTransition>(
              center, inner_radius, /* inner_sphericity */ 0.0, center,
              deformed_radius_,
              /* outer_sphericity */ 1.0,
              static_cast<WedgeTransition::Axis>(gsl::at(axes, j)),
              /* reverse */ true);
        } else {
          transition_func = std::make_unique<WedgeTransition>(
              center, deformed_radius_, /* inner_sphericity */ 1.0, center,
              shape_outer_radius,
              /* outer_sphericity */ 1.0,
              static_cast<WedgeTransition::Axis>(gsl::at(axes, j % 6)));
        }
        gsl::at(shape_maps_, j) = ShapeMap{center,
                                           shape_map_options_->l_max,
                                           shape_map_options_->l_max,
                                           std::move(transition_func),
                                           shape_name,
                                           size_name};
      }
    } else {
      // Shape map transitions from 1 to 0 from the inner radius to the first
      // radial partition or to the outer radius
      deformed_radius_ = inner_radius;
      const double shape_outer_radius =
          radial_partitions.empty() ? outer_radius : radial_partitions.front();
      transition_func =
          std::make_unique<domain::CoordinateMaps::ShapeMapTransitionFunctions::
                               SphereTransition>(inner_radius,
                                                 shape_outer_radius);
      shape_maps_[0] = ShapeMap{center,
                                shape_map_options_->l_max,
                                shape_map_options_->l_max,
                                std::move(transition_func),
                                shape_name,
                                size_name};
    }
  }

  const double outer_shell_inner_radius =
      radial_partitions.empty() ? inner_radius : radial_partitions.back();
  inner_rot_scale_trans_map_ = RotScaleTransMap{
      expansion_map_options_.has_value()
          ? std::optional<std::pair<
                std::string, std::string>>{{expansion_name,
                                            expansion_outer_boundary_name}}
          : std::nullopt,
      rotation_map_options_.has_value()
          ? std::optional<std::string>{rotation_name}
          : std::nullopt,
      translation_map_options_.has_value()
          ? std::optional<std::string>{translation_name}
          : std::nullopt,
      outer_shell_inner_radius,
      outer_radius,
      domain::CoordinateMaps::TimeDependent::RotScaleTrans<
          3>::BlockRegion::Inner};
  if (transition_rot_scale_trans_) {
    if (radial_partitions.empty()) {
      ERROR(
          "Currently at least one radial partition is required to transition "
          "the RotScaleTrans map to zero in the outer shell.");
    }
    transition_rot_scale_trans_map_ = RotScaleTransMap{
        expansion_map_options_.has_value()
            ? std::optional<std::pair<
                  std::string, std::string>>{{expansion_name,
                                              expansion_outer_boundary_name}}
            : std::nullopt,
        rotation_map_options_.has_value()
            ? std::optional<std::string>{rotation_name}
            : std::nullopt,
        translation_map_options_.has_value()
            ? std::optional<std::string>{translation_name}
            : std::nullopt,
        outer_shell_inner_radius,
        outer_radius,
        domain::CoordinateMaps::TimeDependent::RotScaleTrans<
            3>::BlockRegion::Transition};
  }
}

// If you edit any of the functions below, be sure to update the documentation
// in the Sphere domain creator as well as this class' documentation.
TimeDependentMapOptions::MapType<Frame::Distorted, Frame::Inertial>
TimeDependentMapOptions::distorted_to_inertial_map(
    const size_t block_number, const bool is_inner_cube) const {
  const bool block_has_shape_map = shape_map_options_.has_value() and
                                   block_number < (filled_ ? 12 : 6) and
                                   not is_inner_cube;
  if (block_has_shape_map) {
    return std::make_unique<DistortedToInertialComposition>(
        inner_rot_scale_trans_map_);
  } else {
    return nullptr;
  }
}

TimeDependentMapOptions::MapType<Frame::Grid, Frame::Distorted>
TimeDependentMapOptions::grid_to_distorted_map(const size_t block_number,
                                               const bool is_inner_cube) const {
  const bool block_has_shape_map = shape_map_options_.has_value() and
                                   block_number < (filled_ ? 12 : 6) and
                                   not is_inner_cube;
  if (block_has_shape_map) {
    // If the interior is not filled we use the SphereTransition function and
    // build only one shape map at index 0 (see `build_maps` above). Otherwise,
    // we use the Wedge transition function and build a shape map for each
    // direction, so we have to use the block number here to get the correct
    // shape map.
    return std::make_unique<GridToDistortedComposition>(
        gsl::at(shape_maps_, filled_ ? block_number : 0));
  } else {
    return nullptr;
  }
}

TimeDependentMapOptions::MapType<Frame::Grid, Frame::Inertial>
TimeDependentMapOptions::grid_to_inertial_map(const size_t block_number,
                                              const bool is_outer_shell,
                                              const bool is_inner_cube) const {
  const bool block_has_shape_map = shape_map_options_.has_value() and
                                   block_number < (filled_ ? 12 : 6) and
                                   not is_inner_cube;
  if (block_has_shape_map) {
    // If the interior is not filled we use the SphereTransition function and
    // build only one shape map at index 0 (see `build_maps` above). Otherwise,
    // we use the Wedge transition function and build a shape map for each
    // direction, so we have to use the block number here to get the correct
    // shape map.
    return std::make_unique<GridToInertialComposition>(
        gsl::at(shape_maps_, filled_ ? block_number : 0),
        inner_rot_scale_trans_map_);
  } else if (is_outer_shell and transition_rot_scale_trans_) {
    return std::make_unique<GridToInertialSimple>(
        transition_rot_scale_trans_map_);
  } else {
    return std::make_unique<GridToInertialSimple>(inner_rot_scale_trans_map_);
  }
}

bool TimeDependentMapOptions::using_distorted_frame() const {
  // We use shape map options and not the shape map just in case this is called
  // before `build_maps` is called.
  return shape_map_options_.has_value();
}
}  // namespace domain::creators::sphere
