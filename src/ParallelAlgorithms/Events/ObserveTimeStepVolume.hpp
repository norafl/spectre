// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "IO/H5/TensorData.hpp"
#include "Options/String.hpp"
#include "ParallelAlgorithms/Events/ObserveConstantsPerElement.hpp"
#include "Utilities/Serialization/CharmPupable.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
enum class FloatingPointType;
template <size_t VolumeDim>
class Domain;
template <size_t VolumeDim>
class ElementId;
class TimeDelta;
namespace domain {
namespace FunctionsOfTime {
class FunctionOfTime;
}  // namespace FunctionsOfTime
namespace Tags {
template <size_t VolumeDim>
struct Domain;
struct FunctionsOfTime;
template <size_t VolumeDim, typename Frame>
struct MinimumGridSpacing;
}  // namespace Tags
}  // namespace domain
namespace Parallel {
template <typename Metavariables>
class GlobalCache;
}  // namespace Parallel
namespace PUP {
class er;
}  // namespace PUP
namespace Tags {
struct Time;
struct TimeStep;
}  // namespace Tags
/// \endcond

namespace dg::Events {
/*!
 * \brief %Observe the time step in the volume.
 *
 * Observe the time step size in each element.  Each element is output
 * as a single cell with two points per dimension and the observation
 * constant on all those points.
 *
 * Writes volume quantities:
 * - InertialCoordinates (only element corners)
 * - Time step
 * - Slab fraction
 * - Minimum grid spacing
 */
template <size_t VolumeDim>
class ObserveTimeStepVolume : public ObserveConstantsPerElement<VolumeDim> {
 public:
  /// \cond
  explicit ObserveTimeStepVolume(CkMigrateMessage* m);
  using PUP::able::register_constructor;
  WRAPPED_PUPable_decl_template(ObserveTimeStepVolume);  // NOLINT
  /// \endcond

  static constexpr Options::String help =
      "Observe the time step in the volume.";

  ObserveTimeStepVolume() = default;

  ObserveTimeStepVolume(const std::string& subfile_name,
                        ::FloatingPointType coordinates_floating_point_type,
                        ::FloatingPointType floating_point_type);

  using compute_tags_for_observation_box = tmpl::list<>;

  using return_tags = tmpl::list<>;
  using argument_tags =
      tmpl::list<::Tags::Time, ::domain::Tags::FunctionsOfTime,
                 ::domain::Tags::Domain<VolumeDim>, ::Tags::TimeStep,
                 domain::Tags::MinimumGridSpacing<VolumeDim, Frame::Inertial>>;

  template <typename Metavariables, typename ParallelComponent>
  void operator()(const double time,
                  const std::unordered_map<
                      std::string,
                      std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>&
                      functions_of_time,
                  const Domain<VolumeDim>& domain, const TimeDelta& time_step,
                  const double minimum_grid_spacing,
                  Parallel::GlobalCache<Metavariables>& cache,
                  const ElementId<VolumeDim>& element_id,
                  const ParallelComponent* const component,
                  const Event::ObservationValue& observation_value) const {
    std::vector<TensorComponent> components =
        assemble_data(time, functions_of_time, domain, element_id, time_step,
                      minimum_grid_spacing);

    this->observe(components, cache, element_id, component, observation_value);
  }

  bool needs_evolved_variables() const override;

  void pup(PUP::er& p) override;

 private:
  std::vector<TensorComponent> assemble_data(
      double time, const domain::FunctionsOfTimeMap& functions_of_time,
      const Domain<VolumeDim>& domain, const ElementId<VolumeDim>& element_id,
      const TimeDelta& time_step, double minimum_grid_spacing) const;
};
}  // namespace dg::Events
