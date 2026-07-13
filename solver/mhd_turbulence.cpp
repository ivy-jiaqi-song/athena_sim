// Athena++ problem generator for homogeneous, driven MHD turbulence.
//
// This file is kept outside the downloaded Athena++ source. The build helper
// copies it into a disposable build tree as src/pgen/mhd_turbulence.cpp.

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
  const Real b1 = pin->GetOrAddReal("problem", "b1", 1.0);
  const Real b2 = pin->GetOrAddReal("problem", "b2", 0.0);
  const Real b3 = pin->GetOrAddReal("problem", "b3", 0.0);

  if (rho0 <= 0.0 || pressure0 <= 0.0) {
    std::stringstream msg;
    msg << "### FATAL ERROR: rho0 and pressure0 must be positive" << std::endl;
    ATHENA_ERROR(msg);
  }

  // A uniform face-centered field is divergence-free by construction.
  for (int k = ks; k <= ke; ++k) {
    for (int j = js; j <= je; ++j) {
      for (int i = is; i <= ie + 1; ++i) {
        pfield->b.x1f(k, j, i) = b1;
      }
    }
  }
  for (int k = ks; k <= ke; ++k) {
    for (int j = js; j <= je + 1; ++j) {
      for (int i = is; i <= ie; ++i) {
        pfield->b.x2f(k, j, i) = b2;
      }
    }
  }
  for (int k = ks; k <= ke + 1; ++k) {
    for (int j = js; j <= je; ++j) {
      for (int i = is; i <= ie; ++i) {
        pfield->b.x3f(k, j, i) = b3;
      }
    }
  }

  for (int k = ks; k <= ke; ++k) {
    for (int j = js; j <= je; ++j) {
      for (int i = is; i <= ie; ++i) {
        phydro->u(IDN, k, j, i) = rho0;
        phydro->u(IM1, k, j, i) = 0.0;
        phydro->u(IM2, k, j, i) = 0.0;
        phydro->u(IM3, k, j, i) = 0.0;

        if (NON_BAROTROPIC_EOS) {
          const Real gm1 = peos->GetGamma() - 1.0;
          phydro->u(IEN, k, j, i) = pressure0 / gm1
                                    + 0.5 * (b1*b1 + b2*b2 + b3*b3);
        }
      }
    }
  }
}

void Mesh::UserWorkAfterLoop(ParameterInput *pin) {}
