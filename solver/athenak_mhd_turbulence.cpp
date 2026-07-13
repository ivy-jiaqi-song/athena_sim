// AthenaK problem generator for an isothermal Harris Sheet MHD test.
// Installed into a disposable pinned AthenaK source copy by scripts/pipeline.py.

#include <cmath>
#include <cstdlib>
#include <iostream>

#include "athena.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "mhd/mhd.hpp"
#include "particles/particles.hpp"
#include "pgen/pgen.hpp"

#include <Kokkos_Random.hpp>

void ProblemGenerator::UserProblem(ParameterInput *pin, const bool restart) {
  if (restart) return;

  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;
  if (pmbp->pmhd == nullptr) {
    std::cout << "### FATAL ERROR: harris_sheet requires an <mhd> block" << std::endl;
    std::exit(EXIT_FAILURE);
  }

  const Real rho0 = pin->GetOrAddReal("problem", "rho0", 1.0);
  const Real b0_amp = pin->GetOrAddReal("problem", "b0", 1.0);
  const Real guide_b3 = pin->GetOrAddReal("problem", "guide_b3", 0.0);
  const Real sheet_width = pin->GetOrAddReal("problem", "sheet_width", 0.05);
  const Real noise_amplitude = pin->GetOrAddReal("problem", "noise_amplitude", 1.0e-3);
  const Real sound_speed = pin->GetOrAddReal("problem", "sound_speed", 1.0);
  const Real particle_radius = pin->GetOrAddReal("particles", "injection_radius", 0.02);
  const Real particle_velocity = pin->GetOrAddReal("particles", "velocity_scale", 0.0);
  if (rho0 <= 0.0 || sheet_width <= 0.0 || sound_speed <= 0.0 || particle_radius < 0.0) {
    std::cout << "### FATAL ERROR: rho0, sheet_width, and sound_speed must be positive; "
              << "particle injection_radius must be non-negative" << std::endl;
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
  auto &mbsize = pmbp->pmb->mb_size;
  const Real x1min = pmy_mesh_->mesh_size.x1min;
  const Real x1max = pmy_mesh_->mesh_size.x1max;
  const Real x2min = pmy_mesh_->mesh_size.x2min;
  const Real x2max = pmy_mesh_->mesh_size.x2max;
  const Real x3min = pmy_mesh_->mesh_size.x3min;
  const Real x3max = pmy_mesh_->mesh_size.x3max;
  const Real l1 = x1max - x1min;
  const Real l2 = x2max - x2min;
  const Real l3 = x3max - x3min;
  const Real two_pi = 2.0 * std::acos(-1.0);

  par_for("pgen_harris_sheet", DevExeSpace(), 0, nmb - 1,
          ks, ke, js, je, is, ie,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    const Real x1 = mbsize.d_view(m).x1min + (static_cast<Real>(i - is) + 0.5) * mbsize.d_view(m).dx1;
    const Real x2 = mbsize.d_view(m).x2min + (static_cast<Real>(j - js) + 0.5) * mbsize.d_view(m).dx2;
    const Real x3 = mbsize.d_view(m).x3min + (static_cast<Real>(k - ks) + 0.5) * mbsize.d_view(m).dx3;
    const Real b1 = b0_amp * tanh(sin(two_pi * x2 / l2) / sheet_width);
    const Real density = rho0 + 0.5 * (b0_amp*b0_amp - b1*b1) / (sound_speed*sound_speed);
    const Real vx = noise_amplitude * sin(two_pi * x1 / l1) * sin(two_pi * x2 / l2);
    const Real vy = noise_amplitude * cos(two_pi * x1 / l1) * sin(two_pi * x3 / l3);
    const Real vz = noise_amplitude * sin(two_pi * x3 / l3) * cos(two_pi * x2 / l2);

    u0(m, IDN, k, j, i) = density;
    u0(m, IM1, k, j, i) = density * vx;
    u0(m, IM2, k, j, i) = density * vy;
    u0(m, IM3, k, j, i) = density * vz;

    b0.x1f(m, k, j, i) = b1;
    b0.x2f(m, k, j, i) = 0.0;
    b0.x3f(m, k, j, i) = guide_b3;
    if (i == ie) b0.x1f(m, k, j, i + 1) = b1;
    if (j == je) b0.x2f(m, k, j + 1, i) = 0.0;
    if (k == ke) b0.x3f(m, k + 1, j, i) = guide_b3;
  });

  if (pmbp->ppart != nullptr) {
    auto &pr = pmbp->ppart->prtcl_rdata;
    auto &pi = pmbp->ppart->prtcl_idata;
    auto &npart = pmbp->ppart->nprtcl_thispack;
    auto gids = pmbp->gids;
    Kokkos::Random_XorShift64_Pool<> rand_pool64(13579 + pmbp->gids);
    par_for("pgen_harris_sheet_particles", DevExeSpace(), 0, npart - 1,
    KOKKOS_LAMBDA(const int p) {
      auto rand_gen = rand_pool64.get_state();
      const Real dx = particle_radius * (2.0 * rand_gen.frand() - 1.0);
      const Real dy = particle_radius * (2.0 * rand_gen.frand() - 1.0);
      const Real dz = particle_radius * (2.0 * rand_gen.frand() - 1.0);
      pr(IPX, p) = dx;
      pr(IPY, p) = dy;
      pr(IPZ, p) = dz;
      pr(IPVX, p) = particle_velocity * (2.0 * rand_gen.frand() - 1.0);
      pr(IPVY, p) = particle_velocity * (2.0 * rand_gen.frand() - 1.0);
      pr(IPVZ, p) = particle_velocity * (2.0 * rand_gen.frand() - 1.0);
      int owner = gids;
      for (int m = 0; m < nmb; ++m) {
        if (pr(IPX, p) >= mbsize.d_view(m).x1min && pr(IPX, p) < mbsize.d_view(m).x1max &&
            pr(IPY, p) >= mbsize.d_view(m).x2min && pr(IPY, p) < mbsize.d_view(m).x2max &&
            pr(IPZ, p) >= mbsize.d_view(m).x3min && pr(IPZ, p) < mbsize.d_view(m).x3max) {
          owner = gids + m;
        }
      }
      pi(PGID, p) = owner;
      pi(PTAG, p) = p;
      rand_pool64.free_state(rand_gen);
    });
  }
}
