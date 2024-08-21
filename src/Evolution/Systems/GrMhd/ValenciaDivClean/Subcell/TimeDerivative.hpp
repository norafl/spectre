// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <type_traits>

#include "DataStructures/DataBox/AsAccess.hpp"
#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/PrefixHelpers.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/TaggedContainers.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/Structure/Element.hpp"
#include "Domain/Tags.hpp"
#include "Evolution/BoundaryCorrectionTags.hpp"
#include "Evolution/DgSubcell/CartesianFluxDivergence.hpp"
#include "Evolution/DgSubcell/ComputeBoundaryTerms.hpp"
#include "Evolution/DgSubcell/CorrectPackagedData.hpp"
#include "Evolution/DgSubcell/Projection.hpp"
#include "Evolution/DgSubcell/ReconstructionOrder.hpp"
#include "Evolution/DgSubcell/SubcellOptions.hpp"
#include "Evolution/DgSubcell/Tags/CellCenteredFlux.hpp"
#include "Evolution/DgSubcell/Tags/Coordinates.hpp"
#include "Evolution/DgSubcell/Tags/Jacobians.hpp"
#include "Evolution/DgSubcell/Tags/Mesh.hpp"
#include "Evolution/DgSubcell/Tags/OnSubcellFaces.hpp"
#include "Evolution/DgSubcell/Tags/SubcellOptions.hpp"
#include "Evolution/DiscontinuousGalerkin/Actions/NormalCovectorAndMagnitude.hpp"
#include "Evolution/DiscontinuousGalerkin/Actions/PackageDataImpl.hpp"
#include "Evolution/DiscontinuousGalerkin/MortarTags.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/BoundaryCorrections/BoundaryCorrection.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/FiniteDifference/BoundaryConditionGhostData.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/FiniteDifference/Reconstructor.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/FiniteDifference/Tag.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/Fluxes.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/Sources.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/Subcell/ComputeFluxes.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/System.hpp"
#include "NumericalAlgorithms/FiniteDifference/DerivativeOrder.hpp"
#include "NumericalAlgorithms/FiniteDifference/HighOrderFluxCorrection.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "Parallel/Tags/Metavariables.hpp"
#include "PointwiseFunctions/Hydro/Tags.hpp"
#include "Utilities/CallWithDynamicType.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

#include "Utilities/MakeWithValue.hpp"

namespace grmhd::ValenciaDivClean::subcell {
/*!
 * \brief Compute the time derivative on the subcell grid using FD
 * reconstruction.
 */
struct TimeDerivative {
  template <typename DbTagsList>
  static void apply(const gsl::not_null<db::DataBox<DbTagsList>*> box) {
    using evolved_vars_tag = typename System::variables_tag;
    using evolved_vars_tags = typename evolved_vars_tag::tags_list;
    using prim_tags = typename System::primitive_variables_tag::tags_list;
    using recons_prim_tags = tmpl::push_back<
        prim_tags,
        hydro::Tags::LorentzFactorTimesSpatialVelocity<DataVector, 3>>;
    using fluxes_tags = db::wrap_tags_in<::Tags::Flux, evolved_vars_tags,
                                         tmpl::size_t<3>, Frame::Inertial>;

    ASSERT(
        (db::get<::domain::CoordinateMaps::Tags::CoordinateMap<
             3, Frame::Grid, Frame::Inertial>>(*box))
            .is_identity(),
        "Moving mesh is only partly implemented in ValenciaDivClean. If you "
        "need this look at the complete implementation in GhValenciaDivClean. "
        "You will at least need to update the high-order boundary correction "
        "code to include the right normal vectors/Jacobians.");

    const Mesh<3>& subcell_mesh =
        db::get<evolution::dg::subcell::Tags::Mesh<3>>(*box);
    const Mesh<3>& dg_mesh = db::get<domain::Tags::Mesh<3>>(*box);
    ASSERT(
        is_isotropic(subcell_mesh),
        "The subcell/FD mesh must be isotropic for the FD time derivative but "
        "got "
            << subcell_mesh);
    const size_t reconstructed_num_pts =
        (subcell_mesh.extents(0) + 1) *
        subcell_mesh.extents().slice_away(0).product();

    const tnsr::I<DataVector, 3, Frame::ElementLogical>&
        cell_centered_logical_coords =
            db::get<evolution::dg::subcell::Tags::Coordinates<
                3, Frame::ElementLogical>>(*box);
    std::array<double, 3> one_over_delta_xi{};
    for (size_t i = 0; i < 3; ++i) {
      // Note: assumes isotropic extents
      gsl::at(one_over_delta_xi, i) =
          1.0 / (get<0>(cell_centered_logical_coords)[1] -
                 get<0>(cell_centered_logical_coords)[0]);
    }

    // Inverse jacobian, to be projected on faces
    const auto& inv_jacobian_dg =
        db::get<domain::Tags::InverseJacobian<3, Frame::ElementLogical,
                                              Frame::Inertial>>(*box);
    const auto& det_inv_jacobian_dg = db::get<
        domain::Tags::DetInvJacobian<Frame::ElementLogical, Frame::Inertial>>(
        *box);

    // Velocity of the moving mesh on the DG grid, if applicable.
    const std::optional<tnsr::I<DataVector, 3, Frame::Inertial>>&
        mesh_velocity_dg = db::get<domain::Tags::MeshVelocity<3>>(*box);
    const std::optional<Scalar<DataVector>>& div_mesh_velocity =
        db::get<domain::Tags::DivMeshVelocity>(*box);

    const grmhd::ValenciaDivClean::fd::Reconstructor& recons =
        db::get<grmhd::ValenciaDivClean::fd::Tags::Reconstructor>(*box);

    const Element<3>& element = db::get<domain::Tags::Element<3>>(*box);
    const auto fd_derivative_order =
        db::get<evolution::dg::subcell::Tags::SubcellOptions<3>>(*box)
            .finite_difference_derivative_order();
    std::optional<std::array<std::vector<std::uint8_t>, 3>>
        reconstruction_order_data{};
    std::optional<std::array<gsl::span<std::uint8_t>, 3>>
        reconstruction_order{};
    if (static_cast<int>(fd_derivative_order) < 0) {
      reconstruction_order_data = make_array<3>(std::vector<std::uint8_t>(
          (subcell_mesh.extents(0) + 2) * subcell_mesh.extents(1) *
              subcell_mesh.extents(2),
          std::numeric_limits<std::uint8_t>::max()));
      reconstruction_order = std::array<gsl::span<std::uint8_t>, 3>{};
      for (size_t i = 0; i < 3; ++i) {
        gsl::at(reconstruction_order.value(), i) = gsl::make_span(
            gsl::at(reconstruction_order_data.value(), i).data(),
            gsl::at(reconstruction_order_data.value(), i).size());
      }
    }

    const bool element_is_interior = element.external_boundaries().empty();
    constexpr bool subcell_enabled_at_external_boundary =
        std::decay_t<decltype(db::get<Parallel::Tags::Metavariables>(
            *box))>::SubcellOptions::subcell_enabled_at_external_boundary;

    ASSERT(element_is_interior or subcell_enabled_at_external_boundary,
           "Subcell time derivative is called at a boundary element while "
           "using subcell is disabled at external boundaries."
           "ElementID "
               << element.id());

    // Now package the data and compute the correction
    const auto& boundary_correction =
        db::get<evolution::Tags::BoundaryCorrection<System>>(*box);
    using derived_boundary_corrections =
        typename std::decay_t<decltype(boundary_correction)>::creatable_classes;
    std::array<Variables<evolved_vars_tags>, 3> boundary_corrections{};

    // If the element has external boundaries and subcell is enabled for
    // boundary elements, compute FD ghost data with a given boundary condition.
    if constexpr (subcell_enabled_at_external_boundary) {
      if (not element.external_boundaries().empty()) {
        fd::BoundaryConditionGhostData::apply(box, element, recons);
      }
    }

    call_with_dynamic_type<void, derived_boundary_corrections>(
        &boundary_correction, [&](const auto* derived_correction) {
          using DerivedCorrection = std::decay_t<decltype(*derived_correction)>;
          using dg_package_data_temporary_tags =
              typename DerivedCorrection::dg_package_data_temporary_tags;
          using dg_package_data_argument_tags = tmpl::append<
              evolved_vars_tags, recons_prim_tags, fluxes_tags,
              tmpl::remove_duplicates<tmpl::push_back<
                  dg_package_data_temporary_tags,
                  gr::Tags::SpatialMetric<DataVector, 3>,
                  gr::Tags::SqrtDetSpatialMetric<DataVector>,
                  gr::Tags::InverseSpatialMetric<DataVector, 3>,
                  evolution::dg::Actions::detail::NormalVector<3>>>>;
          static_assert(tmpl::size<dg_package_data_argument_tags>::value ==
                            tmpl::size<fd::tags_list_for_reconstruct>::value,
                        "Package data argument tags and tags list for "
                        "reconstruct have different sizes.");
          static_assert(std::is_same_v<dg_package_data_argument_tags,
                                       fd::tags_list_for_reconstruct>,
                        "Package data argument tags and tags list for "
                        "reconstruct are different.");

          // Computed prims and cons on face via reconstruction
          auto package_data_argvars_lower_face = make_array<3>(
              Variables<dg_package_data_argument_tags>(reconstructed_num_pts));
          auto package_data_argvars_upper_face = make_array<3>(
              Variables<dg_package_data_argument_tags>(reconstructed_num_pts));
          // Copy over the face values of the metric quantities.
          using spacetime_vars_to_copy =
              tmpl::list<gr::Tags::Lapse<DataVector>,
                         gr::Tags::Shift<DataVector, 3>,
                         gr::Tags::SpatialMetric<DataVector, 3>,
                         gr::Tags::SqrtDetSpatialMetric<DataVector>,
                         gr::Tags::InverseSpatialMetric<DataVector, 3>>;
          tmpl::for_each<spacetime_vars_to_copy>(
              [&package_data_argvars_lower_face,
               &package_data_argvars_upper_face,
               &spacetime_vars_on_face =
                   db::get<evolution::dg::subcell::Tags::OnSubcellFaces<
                       typename System::flux_spacetime_variables_tag, 3>>(
                       *box)](auto tag_v) {
                using tag = tmpl::type_from<decltype(tag_v)>;
                for (size_t d = 0; d < 3; ++d) {
                  get<tag>(gsl::at(package_data_argvars_lower_face, d)) =
                      get<tag>(gsl::at(spacetime_vars_on_face, d));
                  get<tag>(gsl::at(package_data_argvars_upper_face, d)) =
                      get<tag>(gsl::at(spacetime_vars_on_face, d));
                }
              });

          // Reconstruct data to the face
          call_with_dynamic_type<void, typename grmhd::ValenciaDivClean::fd::
                                           Reconstructor::creatable_classes>(
              &recons, [&box, &package_data_argvars_lower_face,
                        &package_data_argvars_upper_face,
                        &reconstruction_order](const auto& reconstructor) {
                using ReconstructorType =
                    std::decay_t<decltype(*reconstructor)>;
                db::apply<
                    typename ReconstructorType::reconstruction_argument_tags>(
                    [&package_data_argvars_lower_face,
                     &package_data_argvars_upper_face, &reconstructor,
                     &reconstruction_order](const auto&... args) {
                      if constexpr (ReconstructorType::use_adaptive_order) {
                        reconstructor->reconstruct(
                            make_not_null(&package_data_argvars_lower_face),
                            make_not_null(&package_data_argvars_upper_face),
                            make_not_null(&reconstruction_order), args...);
                      } else {
                        (void)reconstruction_order;
                        reconstructor->reconstruct(
                            make_not_null(&package_data_argvars_lower_face),
                            make_not_null(&package_data_argvars_upper_face),
                            args...);
                      }
                    },
                    *box);
              });

          using dg_package_field_tags =
              typename DerivedCorrection::dg_package_field_tags;
          // Allocated outside for loop to reduce allocations
          Variables<dg_package_field_tags> upper_packaged_data{
              reconstructed_num_pts};
          Variables<dg_package_field_tags> lower_packaged_data{
              reconstructed_num_pts};

          // Compute fluxes on faces
          for (size_t i = 0; i < 3; ++i) {
            // Build extents of mesh shifted by half a grid cell in direction i
            const unsigned long& num_subcells_1d = subcell_mesh.extents(0);
            Index<3> face_mesh_extents(std::array<size_t, 3>{
                num_subcells_1d, num_subcells_1d, num_subcells_1d});
            face_mesh_extents[i] = num_subcells_1d + 1;

            auto& vars_upper_face = gsl::at(package_data_argvars_upper_face, i);
            auto& vars_lower_face = gsl::at(package_data_argvars_lower_face, i);
            grmhd::ValenciaDivClean::subcell::compute_fluxes(
                make_not_null(&vars_upper_face));
            grmhd::ValenciaDivClean::subcell::compute_fluxes(
                make_not_null(&vars_lower_face));

            // Add moving mesh corrections to the fluxes, if needed
            std::optional<tnsr::I<DataVector, 3, Frame::Inertial>>
                mesh_velocity_on_face = {};
            if (mesh_velocity_dg.has_value()) {
              // Project mesh velocity on face mesh.
              // Can we get away with only doing the normal component? It
              // is also used in the packaged data...
              mesh_velocity_on_face = tnsr::I<DataVector, 3, Frame::Inertial>{
                  reconstructed_num_pts};
              for (size_t j = 0; j < 3; j++) {
                // j^th component of the velocity on the i^th directed face
                mesh_velocity_on_face.value().get(j) =
                    evolution::dg::subcell::fd::project_to_faces(
                        mesh_velocity_dg.value().get(j), dg_mesh,
                        face_mesh_extents, i);
              }
              tmpl::for_each<evolved_vars_tags>([&vars_upper_face,
                                                 &vars_lower_face,
                                                 &mesh_velocity_on_face](
                                                    auto tag_v) {
                using tag = tmpl::type_from<decltype(tag_v)>;
                using flux_tag =
                    ::Tags::Flux<tag, tmpl::size_t<3>, Frame::Inertial>;
                using FluxTensor = typename flux_tag::type;
                const auto& var_upper = get<tag>(vars_upper_face);
                const auto& var_lower = get<tag>(vars_lower_face);
                auto& flux_upper = get<flux_tag>(vars_upper_face);
                auto& flux_lower = get<flux_tag>(vars_lower_face);
                for (size_t storage_index = 0; storage_index < var_upper.size();
                     ++storage_index) {
                  const auto tensor_index =
                      var_upper.get_tensor_index(storage_index);
                  for (size_t j = 0; j < 3; j++) {
                    const auto flux_storage_index =
                        FluxTensor::get_storage_index(prepend(tensor_index, j));
                    flux_upper[flux_storage_index] -=
                        mesh_velocity_on_face.value().get(j) *
                        var_upper[storage_index];
                    flux_lower[flux_storage_index] -=
                        mesh_velocity_on_face.value().get(j) *
                        var_lower[storage_index];
                  }
                }
              });
            }

            // Normal vectors in curved spacetime normalized by inverse
            // spatial metric. Note that we use the sign convention on
            // the normal vectors to be compatible with DG.
            //
            // Note that these normal vectors are on all faces inside the DG
            // element since there are a bunch of subcells. We don't use the
            // NormalCovectorAndMagnitude tag in the DataBox right now to avoid
            // conflicts with the DG solver. We can explore in the future if
            // it's possible to reuse that allocation.
            //
            // The unnormalized normal vector is
            // n_j = d \xi^{\hat i}/dx^j
            // with "i" the current face.
            tnsr::i<DataVector, 3, Frame::Inertial> lower_outward_conormal{
                reconstructed_num_pts, 0.0};
            for (size_t j = 0; j < 3; j++) {
              lower_outward_conormal.get(j) =
                  evolution::dg::subcell::fd::project_to_faces(
                      inv_jacobian_dg.get(i, j), dg_mesh, face_mesh_extents, i);
            }
            const auto det_inv_jacobian_face =
                evolution::dg::subcell::fd::project_to_faces(
                    get(det_inv_jacobian_dg), dg_mesh, face_mesh_extents, i);

            const Scalar<DataVector> normalization{sqrt(get(
                dot_product(lower_outward_conormal, lower_outward_conormal,
                            get<gr::Tags::InverseSpatialMetric<DataVector, 3>>(
                                vars_upper_face))))};
            for (size_t j = 0; j < 3; j++) {
              lower_outward_conormal.get(j) =
                  lower_outward_conormal.get(j) / get(normalization);
            }

            tnsr::i<DataVector, 3, Frame::Inertial> upper_outward_conormal{
                reconstructed_num_pts, 0.0};
            for (size_t j = 0; j < 3; j++) {
              upper_outward_conormal.get(j) = -lower_outward_conormal.get(j);
            }
            // Note: we probably should compute the normal vector in addition to
            // the co-vector. Not a huge issue since we'll get an FPE right now
            // if it's used by a Riemann solver.

            // Compute the packaged data
            using dg_package_data_projected_tags = tmpl::append<
                evolved_vars_tags, fluxes_tags, dg_package_data_temporary_tags,
                typename DerivedCorrection::dg_package_data_primitive_tags>;
            evolution::dg::Actions::detail::dg_package_data<System>(
                make_not_null(&upper_packaged_data), *derived_correction,
                vars_upper_face, upper_outward_conormal, mesh_velocity_on_face,
                *box, typename DerivedCorrection::dg_package_data_volume_tags{},
                dg_package_data_projected_tags{});

            evolution::dg::Actions::detail::dg_package_data<System>(
                make_not_null(&lower_packaged_data), *derived_correction,
                vars_lower_face, lower_outward_conormal, mesh_velocity_on_face,
                *box, typename DerivedCorrection::dg_package_data_volume_tags{},
                dg_package_data_projected_tags{});

            // Now need to check if any of our neighbors are doing DG,
            // because if so then we need to use whatever boundary data
            // they sent instead of what we computed locally.
            //
            // Note: We could check this beforehand to avoid the extra
            // work of reconstruction and flux computations at the
            // boundaries.
            evolution::dg::subcell::correct_package_data<true>(
                make_not_null(&lower_packaged_data),
                make_not_null(&upper_packaged_data), i, element, subcell_mesh,
                db::get<evolution::dg::Tags::MortarData<3>>(*box), 0);

            // Compute the corrections on the faces. We only need to
            // compute this once because we can just flip the normal
            // vectors then
            gsl::at(boundary_corrections, i).initialize(reconstructed_num_pts);
            evolution::dg::subcell::compute_boundary_terms(
                make_not_null(&gsl::at(boundary_corrections, i)),
                *derived_correction, upper_packaged_data, lower_packaged_data,
                db::as_access(*box),
                typename DerivedCorrection::dg_boundary_terms_volume_tags{});
            // We need to multiply by the normal vector normalization
            gsl::at(boundary_corrections, i) *= get(normalization);
            // Also multiply by determinant of Jacobian, following Eq.(34)
            // of 2109.11645
            gsl::at(boundary_corrections, i) *= 1.0 / det_inv_jacobian_face;
          }
        });

    // Now compute the actual time derivatives.
    using variables_tag = typename System::variables_tag;
    using dt_variables_tag = db::add_tag_prefix<::Tags::dt, variables_tag>;
    const gsl::not_null<typename dt_variables_tag::type*> dt_vars_ptr =
        db::mutate<dt_variables_tag>(
            [](const auto local_dt_vars_ptr) { return local_dt_vars_ptr; },
            box);
    dt_vars_ptr->initialize(subcell_mesh.number_of_grid_points());

    using grmhd_source_tags =
        tmpl::transform<ValenciaDivClean::ComputeSources::return_tags,
                        tmpl::bind<db::remove_tag_prefix, tmpl::_1>>;
    sources_impl(
        dt_vars_ptr, *box, grmhd_source_tags{},
        typename grmhd::ValenciaDivClean::ComputeSources::argument_tags{});

    // Zero GRMHD tags that don't have sources.
    tmpl::for_each<typename variables_tag::tags_list>(
        [&dt_vars_ptr](auto evolved_var_tag_v) {
          using evolved_var_tag = tmpl::type_from<decltype(evolved_var_tag_v)>;
          using dt_tag = ::Tags::dt<evolved_var_tag>;
          auto& dt_var = get<dt_tag>(*dt_vars_ptr);
          for (size_t i = 0; i < dt_var.size(); ++i) {
            if constexpr (not tmpl::list_contains_v<grmhd_source_tags,
                                                    evolved_var_tag>) {
              dt_var[i] = 0.0;
            }
          }
        });

    // Correction to source terms due to moving mesh
    if (div_mesh_velocity.has_value()) {
      const DataVector div_mesh_velocity_subcell =
          evolution::dg::subcell::fd::project(div_mesh_velocity.value().get(),
                                              dg_mesh, subcell_mesh.extents());
      const auto& evolved_vars = db::get<evolved_vars_tag>(*box);

      tmpl::for_each<typename variables_tag::tags_list>(
          [&dt_vars_ptr, &div_mesh_velocity_subcell,
           &evolved_vars](auto evolved_var_tag_v) {
            using evolved_var_tag =
                tmpl::type_from<decltype(evolved_var_tag_v)>;
            using dt_tag = ::Tags::dt<evolved_var_tag>;
            auto& dt_var = get<dt_tag>(*dt_vars_ptr);
            const auto& evolved_var = get<evolved_var_tag>(evolved_vars);
            for (size_t i = 0; i < dt_var.size(); ++i) {
              dt_var[i] -= div_mesh_velocity_subcell * evolved_var[i];
            }
          });
    }

    if (UNLIKELY(fd_derivative_order != ::fd::DerivativeOrder::Two)) {
      ERROR(
          "We don't yet have high-order flux corrections for curved/moving "
          "meshes and the implementation assumes curved/moving meshes. We need "
          "to dot the Cartesian fluxes into the cell-centered "
          "J inv(J)^{hat{i}}_j to get JF^{hat{i}} = J inv(J)^{hat{i}}_j F^j."
          " Some care needs to be taken since we also get F^j from our "
          "neighbors, which leaves the question as to whether to interpolate "
          "the _inertial fluxes_ and then transform or whether to transform "
          "and then interpolate the _densitized logical fluxes_.");
    }
    std::optional<std::array<Variables<evolved_vars_tags>, 3>>
        high_order_corrections{};
    ::fd::cartesian_high_order_flux_corrections(
        make_not_null(&high_order_corrections),

        db::get<evolution::dg::subcell::Tags::CellCenteredFlux<
            evolved_vars_tags, 3>>(*box),
        boundary_corrections, fd_derivative_order,
        db::get<evolution::dg::subcell::Tags::GhostDataForReconstruction<3>>(
            *box),
        subcell_mesh, recons.ghost_zone_size(),
        reconstruction_order.value_or(
            std::array<gsl::span<std::uint8_t>, 3>{}));

    const auto& cell_centered_det_inv_jacobian = db::get<
        evolution::dg::subcell::fd::Tags::DetInverseJacobianLogicalToInertial>(
        *box);
    for (size_t dim = 0; dim < 3; ++dim) {
      const auto& boundary_correction_in_axis =
          high_order_corrections.has_value()
              ? gsl::at(high_order_corrections.value(), dim)
              : gsl::at(boundary_corrections, dim);
      const double inverse_delta = gsl::at(one_over_delta_xi, dim);

      // how to get associated tags for a given variable
      using b_tag = grmhd::ValenciaDivClean::Tags::TildeB<>;
      auto& dt_tilde_b = get<::Tags::dt<b_tag>>(*dt_vars_ptr);
      auto& b_correction = get<b_tag>(boundary_correction_in_axis);

      auto tensor_zeros = make_with_value<
        tnsr::I<DataVector, 3>>(dt_tilde_b, 0.0);
      for (size_t i = 0; i < dt_tilde_b.size(); ++i) {
        evolution::dg::subcell::add_cartesian_flux_divergence(
          make_not_null(&tensor_zeros[i]), inverse_delta,
          get(cell_centered_det_inv_jacobian), b_correction[i],
          subcell_mesh.extents(), dim);
      }

      tmpl::for_each<typename variables_tag::tags_list>(
          [&dt_vars_ptr, &boundary_correction_in_axis,
           &cell_centered_det_inv_jacobian, dim, inverse_delta,
           &subcell_mesh](auto evolved_var_tag_v) {
            using evolved_var_tag =
                tmpl::type_from<decltype(evolved_var_tag_v)>;
            using dt_tag = ::Tags::dt<evolved_var_tag>;
            auto& dt_var = get<dt_tag>(*dt_vars_ptr);
            const auto& var_correction =
              get<evolved_var_tag>(boundary_correction_in_axis);
            for (size_t i = 0; i < dt_var.size(); ++i) {
              evolution::dg::subcell::add_cartesian_flux_divergence(
                  make_not_null(&dt_var[i]), inverse_delta,
                  get(cell_centered_det_inv_jacobian), var_correction[i],
                  subcell_mesh.extents(), dim);
            }/*
            evolution::dg::subcell::add_cartesian_flux_divergence(
                  make_not_null(&zeros), inverse_delta,
                  get(cell_centered_det_inv_jacobian), var_correction[size],
                  subcell_mesh.extents(), dim);*/
            /*
            evolution::dg::subcell::add_cartesian_flux_divergence(
                make_not_null(&dt_var[1]), inverse_delta,
                get(cell_centered_det_inv_jacobian),
                get<grmhd::ValenciaDivClean::Tags::TildeB<>>,
                subcell_mesh.extents(), dim);*/
            // attempt to call cartesian_flux_divergence for non-BC terms
          });
    }

    evolution::dg::subcell::store_reconstruction_order_in_databox(
        box, reconstruction_order);
    /*
    for (size_t i = 0; i < 3; ++i){
      get<::Tags::dt<grmhd::ValenciaDivClean::Tags::TildeB<>>>
        (*dt_vars_ptr).get(i) = 0;
    } // sets time derivative of magnetic field to 0
    */
  }

 private:
  template <typename DtVarsList, typename DbTagsList, typename... SourcedTags,
            typename... ArgsTags>
  static void sources_impl(
      const gsl::not_null<Variables<DtVarsList>*> dt_vars_ptr,
      const db::DataBox<DbTagsList>& box, tmpl::list<SourcedTags...> /*meta*/,
      tmpl::list<ArgsTags...> /*meta*/) {
    grmhd::ValenciaDivClean::ComputeSources::apply(
        get<::Tags::dt<SourcedTags>>(dt_vars_ptr)..., get<ArgsTags>(box)...);
  }
};
}  // namespace grmhd::ValenciaDivClean::subcell
