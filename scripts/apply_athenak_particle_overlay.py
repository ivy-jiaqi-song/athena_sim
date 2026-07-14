#!/usr/bin/env python3
"""Patch pinned AthenaK CUDA tracked-particle output in disposable build trees.

The archived source captures a host stack counter inside a device kernel and
writes tracked records with element offsets where byte offsets are required.
This narrow, hash-guarded overlay fixes those archive bugs without modifying
the external source checkout.
"""

from __future__ import annotations

import hashlib
from pathlib import Path


INPUT_SHA256 = "75622d9f316770f9da640605cfabc4d4d242f97f1edac79fadb21943eb8bbcd2"


def apply_overlay(source: Path) -> str:
    path = source / "src" / "outputs" / "track_prtcl.cpp"
    original = path.read_bytes()
    digest = hashlib.sha256(original).hexdigest()
    if digest != INPUT_SHA256:
        raise RuntimeError(
            f"AthenaK tracked-particle source SHA256 is {digest}; expected {INPUT_SHA256}"
        )
    text = original.decode("utf-8")
    replacements = {
        """  int counter=0;
  int *pcounter = &counter;
  int ntrack_ = ntrack;
  par_for("part_update",DevExeSpace(),0,(npart-1), KOKKOS_LAMBDA(const int p) {
    if (pi(PTAG,p) < ntrack_) {
      int index = Kokkos::atomic_fetch_add(pcounter,1);""":
        """  Kokkos::View<int, DevMemSpace> counter("tracked_particle_counter");
  Kokkos::deep_copy(counter, 0);
  int ntrack_ = ntrack;
  par_for("part_update",DevExeSpace(),0,(npart-1), KOKKOS_LAMBDA(const int p) {
    if (pi(PTAG,p) < ntrack_) {
      int index = Kokkos::atomic_fetch_add(&counter(),1);""",
        """  });
  npout = counter;
  // share number of tracked particles""":
        """  });
  Kokkos::deep_copy(npout, counter);
  // share number of tracked particles""",
        "std::size_t myoffset = header_offset + 6*outpart(p).tag;":
        "std::size_t myoffset = header_offset + 6*sizeof(float)*outpart(p).tag;",
        "partfile.Write_any_type_at_all(&(data[0]),6,myoffset,\"float\")":
        "partfile.Write_any_type_at_all(&(data[6*p]),6,myoffset,\"float\")",
        "partfile.Write_any_type_at(&(data[0]),6,myoffset,\"float\")":
        "partfile.Write_any_type_at(&(data[6*p]),6,myoffset,\"float\")",
    }
    for old, new in replacements.items():
        count = text.count(old)
        expected = 2 if old.startswith("std::size_t myoffset") else 1
        if count != expected:
            raise RuntimeError(f"Expected {expected} occurrence(s) of particle overlay anchor")
        text = text.replace(old, new)
    path.write_text(text, encoding="utf-8")
    return hashlib.sha256(path.read_bytes()).hexdigest()
