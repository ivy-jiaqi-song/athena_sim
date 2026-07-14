// Athena++ problem generator for an isothermal Harris Sheet MHD test.
//
// This file is kept outside the downloaded Athena++ source. The build helper
// copies it into a disposable build tree as src/pgen/mhd_turbulence.cpp.

#include <cmath>
#include <sstream>

#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../eos/eos.hpp"
#include "../field/field.hpp"
#include "../hydro/hydro.hpp"
#include "../mesh/mesh.hpp"
#include "../parameter_input.hpp"

void Mesh::InitUserMeshData(ParameterInput *pin) {
  if (!MAGNETIC_FIELDS_ENABLED) {
    std::stringstream msg;
    msg << "### FATAL ERROR: mhd_turbulence must be configured with -b" << std::endl;
    ATHENA_ERROR(msg);
  }
}

void MeshBlock::ProblemGenerator(ParameterInput *pin) {
  const Real rho0 = pin->GetOrAddReal("problem", "rho0", 1.0);
  const Real pressure0 = pin->GetOrAddReal("problem", "pressure0", 1.0);
  const Real b0 = pin->GetOrAddReal("problem", "b0", 1.0);
  const Real guide_b3 = pin->GetOrAddReal("problem", "guide_b3", 0.0);
  const Real sheet_width = pin->GetOrAddReal("problem", "sheet_width", 0.05);
  const Real noise_amplitude = pin->GetOrAddReal("problem", "noise_amplitude", 1.0e-3);
  const Real sound_speed = pin->GetOrAddReal("problem", "sound_speed", 1.0);
  const Real x1min = pmy_mesh->mesh_size.x1min;
  const Real x1max = pmy_mesh->mesh_size.x1max;
  const Real x2min = pmy_mesh->mesh_size.x2min;
  const Real x2max = pmy_mesh->mesh_size.x2max;
  const Real x3min = pmy_mesh->mesh_size.x3min;
  const Real x3max = pmy_mesh->mesh_size.x3max;
  const Real l1 = x1max - x1min;
  const Real l2 = x2max - x2min;
  const Real l3 = x3max - x3min;
  const Real two_pi = 2.0 * std::acos(-1.0);
  const Real sheet_scale = two_pi * sheet_width / l2;

  if (rho0 <= 0.0 || pressure0 <= 0.0 || sheet_width <= 0.0 || sound_speed <= 0.0) {
    std::stringstream msg;
    msg << "### FATAL ERROR: rho0, pressure0, sheet_width, and sound_speed must be positive"
        << std::endl;
    ATHENA_ERROR(msg);
  }

  // The reversing field is tangential to the sheet and depends only on x2, so
  // the face-centered field remains discretely divergence-free.
  for (int k = ks; k <= ke; ++k) {
    for (int j = js; j <= je; ++j) {
      for (int i = is; i <= ie + 1; ++i) {
        const Real x2 = pcoord->x2v(j);
        pfield->b.x1f(k, j, i) = b0 * std::tanh(std::sin(two_pi * x2 / l2) / sheet_scale);
      }
    }
  }
  for (int k = ks; k <= ke; ++k) {
    for (int j = js; j <= je + 1; ++j) {
      for (int i = is; i <= ie; ++i) {
        pfield->b.x2f(k, j, i) = 0.0;
      }
    }
  }
  for (int k = ks; k <= ke + 1; ++k) {
    for (int j = js; j <= je; ++j) {
      for (int i = is; i <= ie; ++i) {
        pfield->b.x3f(k, j, i) = guide_b3;
      }
    }
  }

  for (int k = ks; k <= ke; ++k) {
    for (int j = js; j <= je; ++j) {
      for (int i = is; i <= ie; ++i) {
        const Real x1 = pcoord->x1v(i);
        const Real x2 = pcoord->x2v(j);
        const Real x3 = pcoord->x3v(k);
        const Real b1 = b0 * std::tanh(std::sin(two_pi * x2 / l2) / sheet_scale);
        const Real density = rho0 + 0.5 * (b0*b0 - b1*b1) / (sound_speed*sound_speed);
        const Real vx = noise_amplitude * std::sin(two_pi * x1 / l1) * std::sin(two_pi * x2 / l2);
        const Real vy = noise_amplitude * std::cos(two_pi * x1 / l1) * std::sin(two_pi * x3 / l3);
        const Real vz = noise_amplitude * std::sin(two_pi * x3 / l3) * std::cos(two_pi * x2 / l2);

        phydro->u(IDN, k, j, i) = density;
        phydro->u(IM1, k, j, i) = density * vx;
        phydro->u(IM2, k, j, i) = density * vy;
        phydro->u(IM3, k, j, i) = density * vz;

        if (NON_BAROTROPIC_EOS) {
          const Real gm1 = peos->GetGamma() - 1.0;
          phydro->u(IEN, k, j, i) = pressure0 / gm1 + 0.5 * density * (vx*vx + vy*vy + vz*vz)
                                    + 0.5 * (b1*b1 + guide_b3*guide_b3);
        }
      }
    }
  }
}

void Mesh::UserWorkAfterLoop(ParameterInput *pin) {}
