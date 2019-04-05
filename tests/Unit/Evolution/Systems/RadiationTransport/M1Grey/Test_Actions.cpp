// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <array>
#include <boost/variant/get.hpp> // IWYU pragma: keep
#include <cstddef>
#include <unordered_map>
#include <utility>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/Tags.hpp"
#include "Evolution/Systems/RadiationTransport/M1Grey/M1Closure.hpp"
#include "Evolution/Systems/RadiationTransport/M1Grey/Tags.hpp"  // IWYU pragma: keep
#include "Evolution/Systems/RadiationTransport/M1Grey/UpdateM1Closure.hpp"  // IWYU pragma: keep
#include "Evolution/Systems/RadiationTransport/Tags.hpp"  // IWYU pragma: keep
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"
#include "tests/Unit/ActionTesting.hpp"

// IWYU pragma: no_forward_declare Tensor

namespace {

template <typename Metavariables>
struct mock_component {
  using metavariables = Metavariables;
  using chare_type = ActionTesting::MockArrayChare;
  using array_index = size_t;
  using Closure = typename RadiationTransport::M1Grey::ComputeM1Closure<
      typename Metavariables::neutrino_species>;
  using action_list = tmpl::list<Actions::UpdateM1Closure>;
  using const_global_cache_tag_list =
      Parallel::get_const_global_cache_tags<action_list>;
  using initial_databox = db::compute_databox_type<tmpl::flatten<
      tmpl::list<typename Closure::return_tags, typename Closure::argument_tags,
                 ::Tags::Coordinates<3, Frame::Inertial>>>>;
};

struct Metavariables {
  using component_list = tmpl::list<mock_component<Metavariables>>;
  using const_global_cache_tag_list = tmpl::list<>;
  using neutrino_species = tmpl::list<neutrinos::ElectronNeutrinos<1>,
                                      neutrinos::HeavyLeptonNeutrinos<0>>;
  enum class Phase { Initialize, Exit };
};
}  // namespace

SPECTRE_TEST_CASE("Unit.RadiationTransport.M1Grey.Actions", "[Unit][M1Grey]") {
  using TupleOfMockDistributedObjects =
      typename ActionTesting::MockRuntimeSystem<
          Metavariables>::TupleOfMockDistributedObjects;
  using component = mock_component<Metavariables>;
  using MockRuntimeSystem = ActionTesting::MockRuntimeSystem<Metavariables>;
  using MockDistributedObjectsTag =
      typename MockRuntimeSystem::template MockDistributedObjectsTag<component>;
  TupleOfMockDistributedObjects dist_objects{};
  const DataVector x{-2.0, -1.0, 0.0, 1.0, 2.0};
  const DataVector y{-2.0, -1.0, 0.0, 1.0, 2.0};
  const DataVector z{-2.0, -1.0, 0.0, 1.0, 2.0};

  // Closure output variables (initialization only,
  // use the same value for all species)
  const DataVector xi{0.5, 0.5, 0.5, 0.5, 0.5};
  const auto tildeP = make_with_value<tnsr::II<DataVector, 3>>(x, 0.5);
  const DataVector J{0.5, 0.5, 0.5, 0.5, 0.5};
  const DataVector Hn{0.5, 0.5, 0.5, 0.5, 0.5};
  const DataVector Hx{0.5, 0.5, 0.5, 0.5, 0.5};
  const DataVector Hy{0.5, 0.5, 0.5, 0.5, 0.5};
  const DataVector Hz{0.5, 0.5, 0.5, 0.5, 0.5};
  // Closure input variables
  // First neutrino species (set to optically thick)
  const DataVector E0{1.0, 1.0, 1.0, 1.0, 1.0};
  const DataVector Sx0{0.0, 0.0, 0.0, 0.0, 0.0};
  const DataVector Sy0{0.0, 0.0, 0.0, 0.0, 0.0};
  const DataVector Sz0{0.0, 0.0, 0.0, 0.0, 0.0};
  // Second neutrino species (set to optically thin)
  const DataVector E1{1.0, 1.0, 1.0, 1.0, 1.0};
  const DataVector Sx1{0.0, 0.0, 0.0, 1.0, 1.0};
  const DataVector Sy1{0.0, 0.0, 1.0, 0.0, 0.0};
  const DataVector Sz1{1.0, 1.0, 0.0, 0.0, 0.0};

  // Fluid and metric variables
  const DataVector vx{0.0, 0.0, 0.0, 0.0, 0.0};
  const DataVector vy{0.0, 0.0, 0.0, 0.0, 0.0};
  const DataVector vz{0.0, 0.0, 0.0, 0.0, 0.0};
  const DataVector W{1.0, 1.0, 1.0, 1.0, 1.0};
  auto metric = make_with_value<tnsr::ii<DataVector, 3>>(x, 0.0);
  auto inverse_metric = make_with_value<tnsr::II<DataVector, 3>>(x, 0.0);
  get<0, 0>(metric) = 1.;
  get<1, 1>(metric) = 1.;
  get<2, 2>(metric) = 1.;
  get<0, 0>(inverse_metric) = 1.;
  get<1, 1>(inverse_metric) = 1.;
  get<2, 2>(inverse_metric) = 1.;

  tuples::get<MockDistributedObjectsTag>(dist_objects)
      .emplace(
          0, ActionTesting::MockDistributedObject<component>{
                 db::create<db::AddSimpleTags<tmpl::flatten<
                     tmpl::list<component::Closure::return_tags,
                                component::Closure::argument_tags,
                                ::Tags::Coordinates<3, Frame::Inertial>>>>>(
                     Scalar<DataVector>{xi}, Scalar<DataVector>{xi}, tildeP,
                     tildeP, Scalar<DataVector>{J}, Scalar<DataVector>{J},
                     Scalar<DataVector>{Hn}, Scalar<DataVector>{Hn},
                     tnsr::i<DataVector, 3, Frame::Inertial>{{{Hx, Hy, Hz}}},
                     tnsr::i<DataVector, 3, Frame::Inertial>{{{Hx, Hy, Hz}}},
                     Scalar<DataVector>{E0}, Scalar<DataVector>{E1},
                     tnsr::i<DataVector, 3, Frame::Inertial>{{{Sx0, Sy0, Sz0}}},
                     tnsr::i<DataVector, 3, Frame::Inertial>{{{Sx1, Sy1, Sz1}}},
                     tnsr::I<DataVector, 3, Frame::Inertial>{{{vx, vy, vz}}},
                     Scalar<DataVector>{W}, metric, inverse_metric,
                     tnsr::I<DataVector, 3, Frame::Inertial>{{{x, y, z}}})});
  MockRuntimeSystem runner{{}, std::move(dist_objects)};
  auto& box = runner.template algorithms<component>()
                  .at(0)
                  .template get_databox<typename component::initial_databox>();
  runner.next_action<component>(0);

  // Check that first species return xi=0 (optically thick)
  const DataVector expected_xi0{0.0, 0.0, 0.0, 0.0, 0.0};
  CHECK_ITERABLE_APPROX(db::get<RadiationTransport::M1Grey::Tags::ClosureFactor<
                            neutrinos::ElectronNeutrinos<1>>>(box)
                            .get(),
                        expected_xi0);
  // Check that second species return xi=1 (optically thin)
  const DataVector expected_xi1{1.0, 1.0, 1.0, 1.0, 1.0};
  CHECK_ITERABLE_APPROX(db::get<RadiationTransport::M1Grey::Tags::ClosureFactor<
                            neutrinos::HeavyLeptonNeutrinos<0>>>(box)
                            .get(),
                        expected_xi1);
}
