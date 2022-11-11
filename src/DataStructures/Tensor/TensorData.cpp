// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "DataStructures/Tensor/TensorData.hpp"

#include <ostream>
#include <pup.h>
#include <pup_stl.h>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "DataStructures/DataVector.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "Parallel/PupStlCpp17.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/StdHelpers.hpp"  // std::vector ostream

TensorComponent::TensorComponent(std::string in_name, DataVector in_data)
    : name(std::move(in_name)), data(std::move(in_data)) {}

TensorComponent::TensorComponent(std::string in_name,
                                 std::vector<float> in_data)
    : name(std::move(in_name)), data(std::move(in_data)) {}

void TensorComponent::pup(PUP::er& p) {
  p | name;
  p | data;
}

std::ostream& operator<<(std::ostream& os, const TensorComponent& t) {
  if (t.data.index() == 0) {
    return os << "(" << t.name << ", " << std::get<DataVector>(t.data) << ")";
  } else if (t.data.index() == 1) {
    return os << "(" << t.name << ", " << std::get<std::vector<float>>(t.data)
              << ")";
  } else {
    ERROR("Unknown index value (" << t.data.index()
                                  << ") in std::variant of tensor component.");
  }
}

bool operator==(const TensorComponent& lhs, const TensorComponent& rhs) {
  return lhs.name == rhs.name and lhs.data == rhs.data;
}

bool operator!=(const TensorComponent& lhs, const TensorComponent& rhs) {
  return not(lhs == rhs);
}

ElementVolumeData::ElementVolumeData(
    std::string element_name_in, std::vector<TensorComponent> components,
    std::vector<size_t> extents_in, std::vector<Spectral::Basis> basis_in,
    std::vector<Spectral::Quadrature> quadrature_in)
    : element_name(std::move(element_name_in)),
      tensor_components(std::move(components)),
      extents(std::move(extents_in)),
      basis(std::move(basis_in)),
      quadrature(std::move(quadrature_in)) {}

void ElementVolumeData::pup(PUP::er& p) {
  p | element_name;
  p | tensor_components;
  p | extents;
  p | quadrature;
  p | basis;
}

bool operator==(const ElementVolumeData& lhs, const ElementVolumeData& rhs) {
  return lhs.element_name == rhs.element_name and
         lhs.tensor_components == rhs.tensor_components and
         lhs.extents == rhs.extents and lhs.quadrature == rhs.quadrature and
         lhs.basis == rhs.basis;
}

bool operator!=(const ElementVolumeData& lhs, const ElementVolumeData& rhs) {
  return not(lhs == rhs);
}
