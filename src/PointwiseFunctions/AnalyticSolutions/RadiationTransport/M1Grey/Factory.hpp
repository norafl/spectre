// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "PointwiseFunctions/AnalyticSolutions/RadiationTransport/M1Grey/ConstantM1.hpp"
#include "Utilities/TMPL.hpp"

namespace RadiationTransport::M1Grey::Solutions {
/// \brief List of all analytic solutions
using all_solutions = tmpl::list<ConstantM1>;
}  // namespace RadiationTransport::M1Grey::Solutions
