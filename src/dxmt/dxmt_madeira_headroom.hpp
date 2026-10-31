#pragma once
/* ml2000: MadeiraCtl op-7 memory headroom query, shared by the D3D9 BC mip
 * clamp (d3d9_device.cpp CreateTexture policy) and the transient-pool reclaim.
 *
 * Query shape mirrors QueryHeadroomMB in
 * dxmt/src/d3d11/d3d11_texture_device.cpp:476-487; madeira_ctl_args comes from
 * dxmt/src/winemetal/winemetal.h:1991-1998 (op 7: len =
 * os_proc_available_memory() bytes, ptr = phys_footprint bytes, ret = 1 when
 * known; ret stays 0 in remote mode or when the reading is unavailable).
 *
 * Units are MiB (bytes >> 20), matching the d3d11 clamp's threshold math, so a
 * headroom reading compares directly against a d3d9.mipClampAutoMB-style
 * threshold. Returns kMadeiraHeadroomUnavailable when the query does not work
 * here: callers must never clamp or reclaim on that reading. The sentinel
 * values match mip_clamp::kHeadroomUnknown/kHeadroomUnavailable in
 * dxmt/src/d3d11/d3d11_mip_clamp_policy.hpp (kept local so src/dxmt does not
 * depend on src/d3d11).
 */

#include <cstdint>
#include <cstring>

#include "winemetal.h"

namespace dxmt {

constexpr int64_t kMadeiraHeadroomUnknown = -1;
constexpr int64_t kMadeiraHeadroomUnavailable = -2;

inline int64_t
QueryMadeiraHeadroomMB(int64_t *footprint_mb) {
  struct madeira_ctl_args a;
  memset(&a, 0, sizeof a);
  a.op = 7;
  MadeiraCtl(&a);
  if (a.ret != 1)
    return kMadeiraHeadroomUnavailable;
  if (footprint_mb != nullptr)
    *footprint_mb = (int64_t)(a.ptr >> 20);
  return (int64_t)(a.len >> 20);
}

} // namespace dxmt
