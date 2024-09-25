// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "ParallelAlgorithms/Events/ObserveTimeStepVolume.hpp"

#include <cstddef>
#include <memory>
#include <pup.h>
#include <string>
#include <unordered_map>
#include <vector>

#include "DataStructures/FloatingPointType.hpp"
#include "IO/H5/TensorData.hpp"
#include "ParallelAlgorithms/Events/ObserveConstantsPerElement.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/Event.hpp"
#include "Time/Time.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"

template <size_t VolumeDim>
class Domain;
template <size_t VolumeDim>
class ElementId;
namespace domain::FunctionsOfTime {
class FunctionOfTime;
}  // namespace domain::FunctionsOfTime

namespace dg::Events {
template <size_t VolumeDim>
ObserveTimeStepVolume<VolumeDim>::ObserveTimeStepVolume(
    const std::string& subfile_name,
    const ::FloatingPointType coordinates_floating_point_type,
    const ::FloatingPointType floating_point_type)
    : ObserveConstantsPerElement<VolumeDim>(
          subfile_name, coordinates_floating_point_type, floating_point_type) {}

template <size_t VolumeDim>
ObserveTimeStepVolume<VolumeDim>::ObserveTimeStepVolume(CkMigrateMessage* m)
    : ObserveConstantsPerElement<VolumeDim>(m) {}

template <size_t VolumeDim>
bool ObserveTimeStepVolume<VolumeDim>::needs_evolved_variables() const {
  return false;
}

template <size_t VolumeDim>
void ObserveTimeStepVolume<VolumeDim>::pup(PUP::er& p) {
  ObserveConstantsPerElement<VolumeDim>::pup(p);
}

template <size_t VolumeDim>
std::vector<TensorComponent> ObserveTimeStepVolume<VolumeDim>::assemble_data(
    const double time,
    const std::unordered_map<
        std::string, std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>&
        functions_of_time,
    const Domain<VolumeDim>& domain, const ElementId<VolumeDim>& element_id,
    const TimeDelta& time_step, const double minimum_grid_spacing) const {
  std::vector<TensorComponent> components = this->allocate_and_insert_coords(
      3, time, functions_of_time, domain, element_id);
  this->add_constant(make_not_null(&components), "Time step",
                     time_step.value());
  this->add_constant(make_not_null(&components), "Slab fraction",
                     time_step.fraction().value());
  this->add_constant(make_not_null(&components), "Minimum grid spacing",
                     minimum_grid_spacing);

  return components;
}

template <size_t VolumeDim>
PUP::able::PUP_ID ObserveTimeStepVolume<VolumeDim>::my_PUP_ID = 0;  // NOLINT

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATE(_, data) template class ObserveTimeStepVolume<DIM(data)>;

GENERATE_INSTANTIATIONS(INSTANTIATE, (1, 2, 3))

#undef INSTANTIATE
#undef DIM
}  // namespace dg::Events
