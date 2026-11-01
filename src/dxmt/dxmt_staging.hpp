#pragma once
#include "Metal.hpp"
#include "dxmt_buffer.hpp"
#include "thread.hpp"
#include <cstdint>
#include <atomic>
#include <queue>

namespace dxmt {

enum class StagingMapResult : uint64_t {
  Mappable = 0,
  Renamable = 0xffffffffffffffff,
  Mapped = 0xfffffffffffffffe,
};

class StagingResource {
public:
  void incRef();
  void decRef();

  void useCopyDestination(uint64_t seq_id);
  void useCopySource(uint64_t seq_id);

  StagingMapResult tryMap(uint64_t coherent_seq_id, bool read, bool write);
  void unmap();

  uint64_t allocate(uint64_t coherent_seq_id);
  void updateImmediateName(uint64_t current_seq_id, uint64_t allocation);

  /* MADEIRA (P4 T5, mem-graphics.md section 7(b)(2)): give the spare pool back.
   *
   * `allocate` only ever recycles a retired buffer whose will_free_at has
   * passed, so a resource mapped while the GPU lags keeps minting a new
   * one and the pool only ever grows to its high-water mark. That is the right
   * policy for a streaming title and the wrong one for a process close to its
   * jetsam line: the retired buffers are pure cache.
   *
   * Releases every GPU-complete retired buffer except kStagingSpareReserve of
   * the most recently retired ones (so steady-state map/unmap does not start
   * minting again), preserving the id -> buffer mapping of everything still
   * live: only ids that are sitting IN the fifo are dropped, and each dropped
   * id's slot is emptied rather than renumbered. Skips the resource entirely
   * when its lock is already held by the caller (this is reached from
   * Buffer::allocate, which StagingResource::allocate calls under that lock).
   *
   * coherent_seq_id is the LIVE completion watermark of the process's command
   * queue (0 when the caller cannot identify one). The instance's own last-seen
   * value alone would be stale for exactly the resource worth trimming -- one
   * that stopped being mapped after a burst -- so its retired buffers would
   * never become eligible. The larger of the two is used: a resource cannot have
   * seen a coherent id its own queue has not signalled. Returns the bytes
   * released. */
  size_t releaseSpareBuffers(uint64_t coherent_seq_id);

  /* Retired buffers kept over a relief pass; enough for the next frame's
   * reuse to hit, not enough to matter to the footprint. */
  enum { kStagingSpareReserve = 1 };

  /* ml1246: rename budget. A rename that cannot reuse a retired buffer adds a
   * whole `length` of shared memory that is never given back while the
   * resource lives. Metro 2033 Redux maps one ~54MB upload staging texture
   * over and over inside a single unflushed seq while loading, and every map
   * allocated (and memcpy'd) a fresh copy: 32 of them, 1.6GB, then jetsam.
   * The caller checks these and waits for the GPU instead, as a native driver
   * does for Map(WRITE) on a staging resource that is still being read. */
  bool canRenameWithoutGrowth(uint64_t coherent_seq_id);
  uint64_t
  gpuBusyDistance(uint64_t coherent_seq_id) const {
    return gpu_occupied_until_finished_seq_id > coherent_seq_id
               ? gpu_occupied_until_finished_seq_id - coherent_seq_id
               : 1;
  }
  /* ml1254: the seq whose completion makes a GPU write visible to a read map */
  uint64_t readableAfterSeq() const { return cpu_coherent_after_finished_seq_id; }
  /* bytes held by buffers beyond each live resource's first one */
  static std::atomic<uint64_t> spare_bytes;

  ~StagingResource();

  void *
  mappedImmediateMemory() {
    return buffer_pool[immediate_name_]->mappedMemory(0);
  }

  void *
  mappedMemory(uint64_t id) {
    return buffer_pool[id]->mappedMemory(0);
  }

  Rc<Buffer> &
  buffer() {
    return buffer_;
  };

  Rc<BufferAllocation>
  allocation(uint64_t id) {
    return buffer_pool[id];
  };

  /**
  readonly
   */
  uint32_t bytesPerRow;
  /**
  readonly
   */
  uint32_t bytesPerImage;
  /**
  readonly
   */
  uint64_t length;

  StagingResource(WMT::Device device, uint64_t length, uint32_t bytes_per_row, uint32_t bytes_per_image);

private:
  struct QueueEntry {
    uint64_t id;
    uint64_t will_free_at;
  };

  Rc<Buffer> buffer_;
  uint64_t immediate_name_;
  std::vector<Rc<BufferAllocation>> buffer_pool;
  /* MADEIRA (P4 T5): pool entries held beyond the first, and the newest
   * coherent sequence id this resource itself handed out. releaseSpareBuffers
   * needs both: the count keeps `spare_bytes` exact across releases that leave
   * holes in the pool, and the sequence id is the one domain the fifo's
   * will_free_at values belong to (a foreign queue's id would NOT be safe to
   * compare against them). */
  size_t spare_count_ = 0;
  uint64_t last_coherent_seq_ = 0;
  std::atomic<uint32_t> refcount_ = {0u};
  std::queue<QueueEntry> fifo;
  dxmt::mutex mutex_;
  bool mapped = false;
  // prevent read from staging before
  uint64_t cpu_coherent_after_finished_seq_id = 0;
  // prevent write to staging before
  uint64_t gpu_occupied_until_finished_seq_id = 0;
  WMT::Device device_;
};

/* ============================================================================
 * MADEIRA (P4 T5, mem-graphics.md section 7(b)(2)): FOOTPRINT-KEYED RELIEF.
 *
 * Every pool below is a "keep it for the next caller" cache: the staging
 * resource spares above, the command queue's three ring allocators, the
 * resource initializer's upload heap, the d3d9 buffer-backing pool and the
 * DynamicBuffer retired-name fifos. Their measured ceilings are large enough
 * to matter at the jetsam line -- init-upload peak 354 MB, staging ring 40 MB,
 * d3d9 pool cap 64 MB (ebb6.txt:22150-22153, d3d9_device.hpp:1529) -- and
 * nothing gave them back early, because an idle device keeps a block for 300
 * later submissions.
 *
 * The trigger is the LIVE footprint, not a budget: the DXGI/D3D9 budget
 * queries are answered once per run (ml1075: "1 queries so far"), so a policy
 * keyed on them is dead code. MadeiraCtl op 7 answers both halves on every
 * call -- len = os_proc_available_memory, ptr = phys_footprint -- which is the
 * number that decides whether the next allocation survives.
 * ==========================================================================*/
constexpr uint32_t kMadeiraReliefHeadroomMB = 1024;

/* op 7 reading in MB; false when the platform cannot answer (remote Metal). */
bool MadeiraMemoryHeadroom(int64_t *headroom_mb, int64_t *footprint_mb);

/* True when the headroom is KNOWN and below kMadeiraReliefHeadroomMB. An
 * unknown reading never reports pressure: relieving on a guess would trade
 * performance for nothing. */
bool MadeiraMemoryPressure();

/* One relief pass, or nothing at all. Cheap to call on a hot allocation path:
 * it rate-limits itself (every 64th call, then at most every 250 ms) before it
 * queries anything. Pass the caller's name for the log line. */
void MadeiraRelieveTransientPools(const char *site);

} // namespace dxmt
