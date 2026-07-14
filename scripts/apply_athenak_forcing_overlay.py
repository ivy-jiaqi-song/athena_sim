#!/usr/bin/env python3
"""Apply the repository-owned AthenaK turbulence forcing overlay.

The archived AthenaK source used by the GPU workflow predates the forcing
design on IAS-Astrophysics/athenak ``turb_fix`` at commit
572f644f3ab3379a32ea2f0bec1658348141dc19.  This overlay backports only the
forcing cadence/integrator behavior and timestep-boundary alignment.  It does
not import that branch's restart implementation or unrelated source changes.
"""

from __future__ import annotations

import hashlib
from pathlib import Path


UPSTREAM_COMMIT = "572f644f3ab3379a32ea2f0bec1658348141dc19"
EXPECTED_INPUTS = {
    "src/srcterms/turb_driver.cpp": "840a8d37e9a18c42cc3e4374b6e4ffd0c668e4d74b5a36b01279cef45b839074",
    "src/srcterms/turb_driver.hpp": "41491a1d95396568d53676b7e3e737b901acb5e29db9286425c21e88c7825042",
    "src/mesh/mesh.cpp": "5fecb389b6912c858defac1436980094ad49e1f9c355b7862cf1cde7f8fa06c2",
    "src/eos/primitive-solver/unit_system.cpp": "2e5dd03cefd8c629a24b7f7757e5262c8241b370dbb216cb3ca13228a2eaf14b",
    "src/eos/primitive-solver/unit_system.hpp": "ca162024bfe96a5abe02f8dcc5602add9e8c912be17a30bd46cd68b373fbb280",
    "src/eos/primitive-solver/piecewise_polytrope.cpp": "6485e394912c2bee5bdc1f69b81e85b7d24eaf6d787ef3e22f642b9bec788ac0",
    "src/eos/primitive-solver/eos_hybrid.cpp": "cac75ea87c536c27d3d54e901a6f4fbcd7e24283f994505ab3fb183f24312d3a",
    "src/eos/primitive-solver/eos_compose.cpp": "9e5791b6b326322f27852311626bd586daf91e7f93f2e36bc82fe9bbbcc6ffd8",
    "src/pgen/unit_tests/gauss_legendre.cpp": "eadd0ea2b768d36dfdf51730756adcce0459ea584f01dcaf603f1902c6e52499",
}


def _replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"AthenaK overlay expected one {label}; found {count}")
    return text.replace(old, new, 1)


def _read_verified(root: Path, relative: str) -> str:
    path = root / relative
    raw = path.read_bytes()
    actual = hashlib.sha256(raw).hexdigest()
    expected = EXPECTED_INPUTS[relative]
    if actual != expected:
        raise RuntimeError(
            f"Refusing forcing overlay: {relative} has SHA256 {actual}, expected {expected}"
        )
    return raw.decode("utf-8")


def _patch_header(text: str) -> str:
    text = _replace_once(
        text,
        "  Real tcorr, dedt;\n  Real expo, exp_prl, exp_prp;\n  int driving_type;",
        "  Real tcorr, dedt, dt_turb_update, next_turb_update;\n"
        "  Real expo, exp_prl, exp_prp;\n"
        "  int driving_type, turb_flag, spect_form, random_seed, n_turb_updates;\n"
        "  Real sol_fraction;",
        "forcing parameter declarations",
    )
    return text


def _patch_driver(text: str) -> str:
    text = _replace_once(
        text,
        "#include <algorithm>\n",
        "#include <algorithm>\n#include <cstdlib>\n",
        "cstdlib include",
    )
    text = _replace_once(
        text,
        "  // driving type\n"
        "  driving_type = pin->GetOrAddInteger(\"turb_driving\", \"driving_type\", 0);",
        "  // driving type and spectral form (2 is the Athena++ power-law mapping)\n"
        "  driving_type = pin->GetOrAddInteger(\"turb_driving\", \"driving_type\", 0);\n"
        "  turb_flag = pin->GetOrAddInteger(\"turb_driving\", \"turb_flag\", 2);\n"
        "  spect_form = pin->GetOrAddInteger(\"turb_driving\", \"spect_form\", 2);",
        "driving type parameters",
    )
    text = _replace_once(
        text,
        "  // correlation time\n"
        "  tcorr = pin->GetOrAddReal(\"turb_driving\", \"tcorr\", 0.0);",
        "  // correlation and innovation-update times\n"
        "  tcorr = pin->GetOrAddReal(\"turb_driving\", \"tcorr\", 0.0);\n"
        "  dt_turb_update = pin->GetOrAddReal(\"turb_driving\", \"dt_turb_update\", tcorr);\n"
        "  sol_fraction = pin->GetOrAddReal(\"turb_driving\", \"sol_fraction\", 1.0);\n"
        "  random_seed = pin->GetOrAddInteger(\"turb_driving\", \"random_seed\", 1);\n"
        "  if (dt_turb_update <= 0.0 || spect_form != 2 ||\n"
        "      sol_fraction < 0.0 || sol_fraction > 1.0) {\n"
        "    std::cout << \"### FATAL ERROR: invalid turbulence forcing controls\" << std::endl;\n"
        "    std::exit(EXIT_FAILURE);\n"
        "  }\n"
        "  if (std::abs(sol_fraction - 1.0) > 1.0e-12) {\n"
        "    std::cout << \"### FATAL ERROR: archived forcing overlay supports sol_fraction=1\"\n"
        "              << std::endl;\n"
        "    std::exit(EXIT_FAILURE);\n"
        "  }",
        "forcing cadence parameters",
    )
    text = _replace_once(
        text,
        "  rstate.idum = -1;",
        "  rstate.idum = -std::max(1, std::abs(random_seed));\n"
        "  next_turb_update = 0.0;\n"
        "  n_turb_updates = 0;",
        "random-state initialization",
    )
    text = _replace_once(
        text,
        "  auto id_init = tl->AddTask(&TurbulenceDriver::InitializeModes, this, start);\n"
        "  tl->AddTask(&TurbulenceDriver::AddForcing, this, id_init);",
        "  tl->AddTask(&TurbulenceDriver::InitializeModes, this, start);",
        "pre-integrator task registration",
    )

    signature = "TaskStatus TurbulenceDriver::InitializeModes(Driver *pdrive, int stage) {"
    start = text.index(signature)
    body = text.index("  Mesh *pm = pmy_pack->pmesh;", start)
    text = text[:body] + (
        "  Mesh *pm = pmy_pack->pmesh;\n"
        "  const Real cadence_eps = 64.0*std::numeric_limits<Real>::epsilon() *\n"
        "                           std::max(static_cast<Real>(1.0), std::abs(pm->time));\n"
        "  if (pm->time + cadence_eps < next_turb_update) return TaskStatus::complete;"
    ) + text[body + len("  Mesh *pm = pmy_pack->pmesh;"):]

    tail_start = text.index("  DvceArray5D<Real> u0, u0_;", start)
    tail_end = text.index("//----------------------------------------------------------------------------------------\n//! \\fn apply forcing", tail_start)
    tail = text[tail_start:tail_end]
    tail = tail.replace("force_tmp_(", "force_(")
    tail = _replace_once(
        tail,
        "  DvceArray5D<Real> u0, u0_;",
        "  // Evolve the retained OU acceleration only on configured update boundaries.\n"
        "  auto force_ = force;\n"
        "  const Real fcorr = (n_turb_updates == 0 || tcorr <= 1.0e-6)\n"
        "                         ? static_cast<Real>(0.0)\n"
        "                         : static_cast<Real>(std::exp(-dt_turb_update/tcorr));\n"
        "  const Real gcorr = std::sqrt(std::max(static_cast<Real>(0.0),\n"
        "      static_cast<Real>(1.0) - fcorr*fcorr));\n"
        "  par_for(\"force_OU_update\", DevExeSpace(),0,nmb-1,0,2,ks,ke,js,je,is,ie,\n"
        "  KOKKOS_LAMBDA(int m, int n, int k, int j, int i) {\n"
        "    force_(m,n,k,j,i) = fcorr*force_(m,n,k,j,i) +\n"
        "                         gcorr*force_tmp_(m,n,k,j,i);\n"
        "  });\n\n"
        "  DvceArray5D<Real> u0, u0_;",
        "OU retained-force update",
    )
    tail = tail.replace("  t1 = std::max(t1, 1.0e-20);\n", "")
    tail = tail.replace(
        "std::max(t0, 1.0e-20)",
        "std::max(t0, static_cast<Real>(1.0e-20))",
    )
    tail = _replace_once(
        tail,
        "  return TaskStatus::complete;\n}\n\n",
        "  ++n_turb_updates;\n"
        "  next_turb_update = static_cast<Real>(\n"
        "      std::floor((pm->time + cadence_eps)/dt_turb_update) + 1.0) * dt_turb_update;\n"
        "  return TaskStatus::complete;\n}\n\n",
        "forcing update completion",
    )
    text = text[:tail_start] + tail + text[tail_end:]

    add_start = text.index("TaskStatus TurbulenceDriver::AddForcing(Driver *pdrive, int stage) {")
    add_end = len(text)
    add = text[add_start:add_end]
    add = _replace_once(
        add,
        "  Real dt = pm->dt;\n"
        "  Real fcorr, gcorr;\n"
        "  if (tcorr <= 1e-6) {  // use whitenoise\n"
        "    fcorr = 0.0;\n"
        "    gcorr = 1.0;\n"
        "  } else {\n"
        "    fcorr = std::exp(-dt/tcorr);\n"
        "    gcorr = std::sqrt(1.0 - fcorr*fcorr);\n"
        "  }",
        "  Real dt = pm->dt;\n"
        "  Real bdt = pdrive->beta[stage-1] * dt;",
        "stage timestep",
    )
    add = _replace_once(add, "  auto force_tmp_ = force_tmp;\n\n", "", "temporary force alias")
    ou_start = add.index("  par_for(\"force_OU_process\"")
    push_start = add.index("  par_for(\"push\"", ou_start)
    add = add[:ou_start] + add[push_start:]
    for old, new in (
        ("Fv*den*dt", "Fv*den*bdt"),
        ("den*v1*dt", "den*v1*bdt"),
        ("den*v2*dt", "den*v2*bdt"),
        ("den*v3*dt", "den*v3*bdt"),
    ):
        add = add.replace(old, new)
    text = text[:add_start] + add
    return text


def _patch_mesh(text: str) -> str:
    text = _replace_once(
        text,
        "#include <cinttypes>\n",
        "#include <cinttypes>\n#include <cmath>\n",
        "cmath include",
    )
    text = _replace_once(
        text,
        '#include "srcterms/srcterms.hpp"',
        '#include "srcterms/srcterms.hpp"\n#include "srcterms/turb_driver.hpp"',
        "turbulence driver include",
    )
    text = _replace_once(
        text,
        "  // limit last time step to stop at tlim *exactly*\n"
        "  if ( (time < tlim) && ((time + dt) > tlim) ) {dt = tlim - time;}",
        "  // Land exactly on OU innovation boundaries so one innovation is generated per\n"
        "  // dt_turb_update and the acceleration is fixed across all RK sub-stages.\n"
        "  if (pmb_pack->pturb != nullptr) {\n"
        "    const Real cadence = pmb_pack->pturb->dt_turb_update;\n"
        "    const Real eps = 64.0*std::numeric_limits<Real>::epsilon() *\n"
        "                     std::max(static_cast<Real>(1.0), std::abs(time));\n"
        "    const Real boundary = static_cast<Real>(\n"
        "        std::floor((time + eps)/cadence) + 1.0)*cadence;\n"
        "    if (boundary > time + eps) dt = std::min(dt, boundary - time);\n"
        "  }\n\n"
        "  // limit last time step to stop at tlim *exactly*\n"
        "  if ( (time < tlim) && ((time + dt) > tlim) ) {dt = tlim - time;}",
        "timestep limit block",
    )
    return text


def _patch_unit_initializers(text: str) -> str:
    """Make aggregate unit constants explicit Real conversions for FP32 nvcc."""
    result: list[str] = []
    inside = False
    wrapped = 0
    for line in text.splitlines(keepends=True):
        stripped = line.strip()
        if stripped.startswith("return UnitSystem{") or stripped.startswith("static UnitSystem CGS{"):
            inside = True
            result.append(line)
            continue
        if inside and stripped == "};":
            inside = False
            result.append(line)
            continue
        if inside and stripped and not stripped.startswith("//"):
            newline = "\n" if line.endswith("\n") else ""
            body = line[:-1] if newline else line
            code, marker, comment = body.partition("//")
            if code.rstrip().endswith(","):
                indent = code[: len(code) - len(code.lstrip())]
                expression = code.strip()[:-1]
                body = f"{indent}static_cast<Real>({expression}),"
                if marker:
                    body += f" //{comment}"
                line = body + newline
                wrapped += 1
        result.append(line)
    if wrapped < 13:
        raise RuntimeError(f"Expected FP32 unit initializers; wrapped only {wrapped}")
    return "".join(result)


def _patch_table_reader_pointers(text: str) -> str:
    count = text.count("Real * table_")
    if count == 0:
        raise RuntimeError("Expected TableReader pointers for FP32 compatibility")
    return text.replace("Real * table_", "double * table_")


def _patch_piecewise_polytrope(text: str) -> str:
    text = _replace_once(
        text, "  double densities[MAX_PIECES+1];", "  Real densities[MAX_PIECES+1];",
        "piecewise density array",
    )
    return _replace_once(
        text, "  double gammas[MAX_PIECES+1];", "  Real gammas[MAX_PIECES+1];",
        "piecewise gamma array",
    )


def _patch_gauss_legendre(text: str) -> str:
    return _replace_once(
        text,
        "  double ylmR1,ylmI1,ylmR2,ylmI2;",
        "  Real ylmR1,ylmI1,ylmR2,ylmI2;",
        "spherical-harmonic output storage",
    )


def apply_overlay(source_root: str | Path) -> str:
    root = Path(source_root)
    originals = {relative: _read_verified(root, relative) for relative in EXPECTED_INPUTS}
    patched = {
        "src/srcterms/turb_driver.cpp": _patch_driver(originals["src/srcterms/turb_driver.cpp"]),
        "src/srcterms/turb_driver.hpp": _patch_header(originals["src/srcterms/turb_driver.hpp"]),
        "src/mesh/mesh.cpp": _patch_mesh(originals["src/mesh/mesh.cpp"]),
        "src/eos/primitive-solver/unit_system.cpp": _patch_unit_initializers(
            originals["src/eos/primitive-solver/unit_system.cpp"]
        ),
        "src/eos/primitive-solver/unit_system.hpp": _patch_unit_initializers(
            originals["src/eos/primitive-solver/unit_system.hpp"]
        ),
        "src/eos/primitive-solver/piecewise_polytrope.cpp": _patch_piecewise_polytrope(
            originals["src/eos/primitive-solver/piecewise_polytrope.cpp"]
        ),
        "src/eos/primitive-solver/eos_hybrid.cpp": _patch_table_reader_pointers(
            originals["src/eos/primitive-solver/eos_hybrid.cpp"]
        ),
        "src/eos/primitive-solver/eos_compose.cpp": _patch_table_reader_pointers(
            originals["src/eos/primitive-solver/eos_compose.cpp"]
        ),
        "src/pgen/unit_tests/gauss_legendre.cpp": _patch_gauss_legendre(
            originals["src/pgen/unit_tests/gauss_legendre.cpp"]
        ),
    }
    digest = hashlib.sha256()
    for relative in sorted(patched):
        raw = patched[relative].encode("utf-8")
        (root / relative).write_bytes(raw)
        digest.update(relative.encode("utf-8") + b"\0" + raw)
    return digest.hexdigest()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_root", type=Path)
    args = parser.parse_args()
    print(apply_overlay(args.source_root))

