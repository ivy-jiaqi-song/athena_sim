// AthenaK problem generator for homogeneous driven isothermal MHD turbulence.
// Installed into a disposable pinned AthenaK source copy by scripts/pipeline.py.

#include <cstdlib>
#include <iostream>

#include "athena.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "mhd/mhd.hpp"
#include "pgen/pgen.hpp"

void ProblemGenerator::UserProblem(ParameterInput *pin, const bool restart) {
  if (restart) return;

  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;
  if (pmbp->pmhd == nullptr) {
    std::cout << "### FATAL ERROR: mhd_turbulence requires an <mhd> block" << std::endl;
    std::exit(EXIT_FAILURE);
  }

  const Real rho0 = pin->GetOrAddReal("problem", "rho0", 1.0);
  const Real b1 = pin->GetOrAddReal("problem", "b1", 1.0);
  const Real b2 = pin->GetOrAddReal("problem", "b2", 0.0);
  const Real b3 = pin->GetOrAddReal("problem", "b3", 0.0);
  if (rho0 <= 0.0) {
    std::cout << "### FATAL ERROR: <problem>/rho0 must be positive" << std::endl;
    std::exit(EXIT_FAILURE);
  }

  auto &indcs = pmy_mesh_->mb_indcs;
  const int is = indcs.is;
  const int ie = indcs.ie;
  const int js = indcs.js;
  const int je = indcs.je;
  const int ks = indcs.ks;
  const int ke = indcs.ke;
  const int nmb = pmbp->nmb_thispack;
  auto &u0 = pmbp->pmhd->u0;
  auto &b0 = pmbp->pmhd->b0;

  // Fill every active cell and all upper faces. A spatially uniform face field
  // has exactly zero discrete divergence, including across MeshBlock boundaries.
  par_for("pgen_mhd_turbulence", DevExeSpace(), 0, nmb - 1,
          ks, ke, js, je, is, ie,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    u0(m, IDN, k, j, i) = rho0;
    u0(m, IM1, k, j, i) = 0.0;
    u0(m, IM2, k, j, i) = 0.0;
    u0(m, IM3, k, j, i) = 0.0;

    b0.x1f(m, k, j, i) = b1;
    b0.x2f(m, k, j, i) = b2;
    b0.x3f(m, k, j, i) = b3;
    if (i == ie) b0.x1f(m, k, j, i + 1) = b1;
    if (j == je) b0.x2f(m, k, j + 1, i) = b2;
    if (k == ke) b0.x3f(m, k + 1, j, i) = b3;
  });
}
