#include "dxmt_staging.hpp"
#include "dxmt_mem_census.hpp"
#include "log/log.hpp"
#include "winemetal.h"
#include <algorithm>
#include <cassert>
#include <chrono>
#include <mutex>
#include <vector>

namespace dxmt {

std::atomic<uint64_t> StagingResource::spare_bytes = {0u};

/* MADEIRA (P4 T5): every live StagingResource, so the relief pass can ask each
 * one to give its spares back. Registration is ctor/dtor work; the pass runs on
 * whichever thread hit the allocation choke point, so the list is guarded and
 * the whole walk holds it -- unregistration therefore cannot race a walk into a
 * freed instance (unregister happens in decRef at the last reference, before
 * the delete). Instances whose own lock is busy are skipped by their
 * try_lock, which is also what keeps this safe when the pass is reached from
 * inside StagingResource::allocate (that frame holds the lock and called
 * Buffer::allocate, which is the choke point). */
namespace {
std::mutex g_staging_registry_mutex;
std::vector<StagingResource *> g_staging_registry;
} // namespace

StagingResource::~StagingResource() {
  {
    std::lock_guard<std::mutex> lock(g_staging_registry_mutex);
    auto &list = g_staging_registry;
    for (auto it = list.begin(); it != list.end(); ++it)
      if (*it == this) {
        list.erase(it);
        break;
      }
  }
  if (spare_count_)
    spare_bytes.fetch_sub(spare_count_ * length, std::memory_order_relaxed);
}

bool
StagingResource::canRenameWithoutGrowth(uint64_t coherent_seq_id) {
  std::lock_guard<dxmt::mutex> lock(mutex_);
  return !fifo.empty() && fifo.front().will_free_at <= coherent_seq_id;
}

StagingResource::StagingResource(
  WMT::Device device, uint64_t length, uint32_t bytes_per_row, uint32_t bytes_per_image
) :
    bytesPerRow(bytes_per_row),
    bytesPerImage(bytes_per_image),
    length(length),
    device_(device) {
  buffer_ = new Buffer(length, device);
  immediate_name_ = allocate(0);
  buffer_->rename(allocation(0));
  {
    std::lock_guard<std::mutex> lock(g_staging_registry_mutex);
    g_staging_registry.push_back(this);
  }
}

void
StagingResource::incRef() {
  refcount_.fetch_add(1u, std::memory_order_acquire);
};

void
StagingResource::decRef() {
  if (refcount_.fetch_sub(1u, std::memory_order_release) == 1u)
    delete this;
};

void
StagingResource::useCopyDestination(uint64_t seq_id) {
  assert(!mapped && "write to staging resource while being mapped");
  // coherent read-after-write
  cpu_coherent_after_finished_seq_id = std::max(seq_id, cpu_coherent_after_finished_seq_id);
  // coherent write-after-write
  gpu_occupied_until_finished_seq_id = std::max(seq_id, gpu_occupied_until_finished_seq_id);
}
void
StagingResource::useCopySource(uint64_t seq_id) {
  assert(!mapped && "read from staging resource while being mapped");
  gpu_occupied_until_finished_seq_id = std::max(seq_id, gpu_occupied_until_finished_seq_id);
}

StagingMapResult
StagingResource::tryMap(uint64_t coherent_seq_id, bool read, bool write) {
  if (mapped)
    return StagingMapResult::Mapped;
  if (read && coherent_seq_id < cpu_coherent_after_finished_seq_id)
    return StagingMapResult(cpu_coherent_after_finished_seq_id - coherent_seq_id);
  if (write && coherent_seq_id < gpu_occupied_until_finished_seq_id) {
    if (coherent_seq_id >= cpu_coherent_after_finished_seq_id)
      return StagingMapResult::Renamable;
    return StagingMapResult(gpu_occupied_until_finished_seq_id - coherent_seq_id);
  }
  return StagingMapResult::Mappable;
}

void
StagingResource::unmap() {
  mapped = false;
}

uint64_t
StagingResource::allocate(uint64_t coherent_seq_id) {
  std::lock_guard<dxmt::mutex> lock(mutex_);
  if (coherent_seq_id > last_coherent_seq_)
    last_coherent_seq_ = coherent_seq_id;
  uint64_t ret = ~0ull;
  for (;;) {
    if (fifo.empty()) {
      break;
    }
    auto entry = fifo.front();
    if (entry.will_free_at > coherent_seq_id) {
      break;
    }
    ret = entry.id;
    fifo.pop();
    break;
  }
  if (ret == ~0ull) {
    Flags<BufferAllocationFlag> flags;
#ifdef __i386__
    flags.set(BufferAllocationFlag::CpuPlaced);
#endif
    if (!buffer_pool.empty()) {
      spare_bytes.fetch_add(length, std::memory_order_relaxed);
      spare_count_++;
    }
    buffer_pool.push_back(buffer_->allocate(flags, "StagingResource::pool"));
    ret = buffer_pool.size() - 1;
  }
  return ret;
}

size_t
StagingResource::releaseSpareBuffers(uint64_t coherent_seq_id) {
  /* try_lock, not lock: see the registry note above. A busy resource is skipped,
   * a locked one would deadlock this pass. */
  std::unique_lock<dxmt::mutex> lock(mutex_, std::try_to_lock);
  if (!lock.owns_lock())
    return 0;
  const uint64_t coherent = std::max(coherent_seq_id, last_coherent_seq_);
  if (coherent == 0 || fifo.size() <= (size_t)kStagingSpareReserve)
    return 0;
  /* updateImmediateName stamps will_free_at with the sequence it retires at and
   * the fifo is FIFO, so the GPU-complete (eligible) entries are its prefix --
   * exactly the property that makes dropping from the front safe. */
  size_t eligible = 0;
  {
    std::queue<QueueEntry> probe = fifo;
    while (!probe.empty() && probe.front().will_free_at <= coherent) {
      eligible++;
      probe.pop();
    }
  }
  if (eligible <= (size_t)kStagingSpareReserve)
    return 0;
  size_t drop = eligible - (size_t)kStagingSpareReserve;
  size_t freed = 0, counted = 0;
  while (drop-- && !fifo.empty()) {
    const QueueEntry entry = fifo.front();
    if (entry.will_free_at > coherent)   /* never drop a fenced buffer */
      break;
    fifo.pop();
    /* Empty the slot instead of renumbering: every other id stays valid, and a
     * dropped id cannot be handed out again because it just left the fifo. */
    buffer_pool[entry.id] = nullptr;
    freed += length;
    /* Only the entries allocate() charged spare_bytes for may be subtracted:
     * the first pool entry (id 0) was never counted, and subtracting it would
     * wrap the counter, which is the ml1246 rename budget's input. */
    if (entry.id != 0 && spare_count_) {
      spare_count_--;
      counted += length;
    }
  }
  if (counted)
    spare_bytes.fetch_sub(counted, std::memory_order_relaxed);
  /* A released slot is empty and nothing references it (its id left the fifo
   * above, and the immediate name is never a fifo entry), so the trailing run of
   * empties can go: without this the vector would keep one 8-byte hole per
   * buffer ever released for the rest of the session. Interior holes stay,
   * because erasing one would renumber every id above it. */
  while (!buffer_pool.empty() && !buffer_pool.back())
    buffer_pool.pop_back();
  return freed;
}

void
StagingResource::updateImmediateName(uint64_t current_seq_id, uint64_t allocation) {
  std::lock_guard<dxmt::mutex> lock(mutex_);
  fifo.push(QueueEntry{.id = immediate_name_, .will_free_at = current_seq_id});
  immediate_name_ = allocation;
  cpu_coherent_after_finished_seq_id = 0;
  gpu_occupied_until_finished_seq_id = 0;
}

/* ---------------------------------------------------------------------------
 * MADEIRA (P4 T5): the relief pass itself. Declared in dxmt_staging.hpp.
 * ------------------------------------------------------------------------- */

/* Defined in dxmt_command_queue.cpp (it owns the queue registry) and
 * dxmt_dynamic.cpp (it owns the DynamicBuffer registry); both return the bytes
 * they released. The ring release also hands back the queue's live completion
 * watermark, which the other two trims need to test their fifos. */
size_t ReleaseCommandQueueRingSpares(uint64_t *out_coherent_seq);
size_t TrimInitializerHeapSpares();
size_t TrimDynamicBufferSpares(uint64_t coherent_seq_id);

bool
MadeiraMemoryHeadroom(int64_t *headroom_mb, int64_t *footprint_mb) {
  struct madeira_ctl_args a = {};
  a.op = 7;
  MadeiraCtl(&a);
  if (a.ret != 1)
    return false;
  if (headroom_mb)
    *headroom_mb = (int64_t)(a.len >> 20);
  if (footprint_mb)
    *footprint_mb = (int64_t)(a.ptr >> 20);
  return true;
}

bool
MadeiraMemoryPressure() {
  int64_t headroom_mb = 0;
  if (!MadeiraMemoryHeadroom(&headroom_mb, nullptr))
    return false;
  return headroom_mb < (int64_t)kMadeiraReliefHeadroomMB;
}

void
MadeiraRelieveTransientPools(const char *site) {
  /* Two gates before the query, because Buffer::allocate is the hot path: a
   * relaxed counter first (one increment), then the clock. */
  static std::atomic<uint32_t> calls{0};
  constexpr uint32_t kCheckEvery = 64;
  constexpr uint64_t kMinIntervalMs = 250;
  if (calls.fetch_add(1, std::memory_order_relaxed) % kCheckEvery)
    return;

  /* Not re-entrant: the pass is reached from allocation paths that a release
   * below could, in principle, drive again. */
  static thread_local bool in_pass = false;
  if (in_pass)
    return;
  in_pass = true;

  using clock = std::chrono::steady_clock;
  const uint64_t now_ms = (uint64_t)std::chrono::duration_cast<std::chrono::milliseconds>(
                              clock::now().time_since_epoch())
                              .count();
  static std::atomic<uint64_t> last_pass_ms{0};
  static std::atomic<bool> pressure_logged{false};

  int64_t headroom_mb = 0, footprint_mb = 0;
  if (!MadeiraMemoryHeadroom(&headroom_mb, &footprint_mb)) {
    if (!pressure_logged.exchange(true)) {
      ERR("[relief] mlP4 memory headroom unavailable here (MadeiraCtl op 7); "
          "transient pools keep their normal retention");
    }
    in_pass = false;
    return;
  }

  if (headroom_mb >= (int64_t)kMadeiraReliefHeadroomMB) {
    /* Log each exit from the pressure region once, so a run shows whether the
     * reclaims below ever had headroom to work with. */
    if (pressure_logged.exchange(false)) {
      ERR("[relief] mlP4 headroom=", headroom_mb, " MB footprint=", footprint_mb,
          " MB: recovering, transient pools keep their normal retention");
    }
    in_pass = false;
    return;
  }

  if (!pressure_logged.exchange(true)) {
    ERR("[relief] mlP4 headroom=", headroom_mb, " MB footprint=", footprint_mb,
        " MB is under ", kMadeiraReliefHeadroomMB,
        " MB: releasing transient pool spares on allocation");
  }

  const uint64_t last = last_pass_ms.load(std::memory_order_relaxed);
  if (last && now_ms - last < kMinIntervalMs) {
    in_pass = false;
    return;
  }
  last_pass_ms.store(now_ms, std::memory_order_relaxed);

  const uint64_t staging_before = g_mem_census.live[MEMOWN_STAGING_RING].load(std::memory_order_relaxed);
  const uint64_t init_before = g_mem_census.live[MEMOWN_INIT_UPLOAD].load(std::memory_order_relaxed);

  uint64_t coherent_seq = 0;
  size_t ring_bytes = ReleaseCommandQueueRingSpares(&coherent_seq);
  size_t init_bytes = TrimInitializerHeapSpares();
  /* Both callers measure what their own release actually dropped; the census
   * delta is the cross-check that also catches a block freed by side effect. */
  {
    const uint64_t after = g_mem_census.live[MEMOWN_STAGING_RING].load(std::memory_order_relaxed);
    if (staging_before > after && (staging_before - after) > ring_bytes)
      ring_bytes = (size_t)(staging_before - after);
    const uint64_t init_after = g_mem_census.live[MEMOWN_INIT_UPLOAD].load(std::memory_order_relaxed);
    if (init_before > init_after && (init_before - init_after) > init_bytes)
      init_bytes = (size_t)(init_before - init_after);
  }

  size_t staging_bytes = 0;
  {
    std::lock_guard<std::mutex> lock(g_staging_registry_mutex);
    for (auto *resource : g_staging_registry)
      staging_bytes += resource->releaseSpareBuffers(coherent_seq);
  }
  const size_t dyn_bytes = TrimDynamicBufferSpares(coherent_seq);

  const size_t total = staging_bytes + dyn_bytes + init_bytes + ring_bytes;
  if (!total) {
    in_pass = false;
    return;
  }
  ERR("[relief] mlP4 site=", site, " headroom=", headroom_mb, " MB footprint=", footprint_mb,
      " MB freed=", total >> 10, " KiB (staging-spares=", staging_bytes >> 10,
      " dyn-eligible=", dyn_bytes >> 10, " init-upload-heap=", init_bytes >> 10,
      " staging-ring=", ring_bytes >> 10, " KiB)");
  in_pass = false;
}

} // namespace dxmt
