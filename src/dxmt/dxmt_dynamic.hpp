#pragma once
#include "dxmt_buffer.hpp"
#include "dxmt_texture.hpp"
#include <deque>

namespace dxmt {

class DynamicBuffer {
public:
  void incRef();
  void decRef();

  // out_minted_fresh, when given, reports whether the FIFO had nothing retired
  // to hand back and a new allocation had to be minted. Worth distinguishing
  // because the two outcomes differ in CONTENT, not just in cost: a recycled
  // allocation still holds the bytes it was last written with, while a fresh
  // one comes up zeroed. Anything that reads such a buffer before its upload
  // lands therefore renders last frame's data in the recycled case and nothing
  // at all in the fresh case, so the fresh rate bounds how often that class of
  // mistake can be visible.
  Rc<BufferAllocation> allocate(uint64_t coherent_seq_id, bool *out_minted_fresh = nullptr);
  void updateImmediateName(uint64_t current_seq_id, Rc<BufferAllocation> &&allocation, uint32_t suballocation, bool owned_by_command_list);
  void recycle(uint64_t current_seq_id, Rc<BufferAllocation> &&allocation);
  uint32_t nextSuballocation();

  /* MADEIRA (P4 T5, mem-graphics.md section 7(b)(2)): release the retired names
   * the GPU has finished with, down to kTrimReserveEligible instead of the
   * allocate-time working reserve.
   *
   * allocate() only trims when it is about to mint a replacement, so a fifo
   * that stops being asked for anything keeps its eligible entries for the life
   * of the instance. Under memory pressure those entries are cache, and this is
   * the trim the census already counts (g_trim_total_n/g_trim_total_bytes).
   * Trims against THIS instance's newest coherent sequence id: will_free_at
   * values only mean anything inside the queue that produced them, so a
   * foreign (possibly larger) id could drop an entry still in GPU use.
   * coherent_seq_id is the caller's live watermark for that same queue (0 when
   * it cannot identify one); the larger of it and the instance's own last-seen
   * value is used, because a fifo that stopped being asked for anything would
   * otherwise keep an eligible tail nobody ever marks as complete.
   * Returns the bytes released. */
  size_t trimEligible(uint64_t coherent_seq_id);

  enum { kTrimReserveEligible = 8 };

  /* Cross-instance entry point for the relief pass. */
  static size_t trimAllEligible(uint64_t coherent_seq_id);

  Rc<BufferAllocation>
  immediateName() {
    return name_;
  }

  uint32_t
  immediateSuballocation() {
    return name_suballocation_;
  }

  void *
  immediateMappedMemory() {
    return name_->mappedMemory(name_suballocation_);
  }

  DynamicBuffer(Buffer *buffer, Flags<BufferAllocationFlag> flags);

  struct QueueEntry {
    Rc<BufferAllocation> allocation;
    uint64_t will_free_at;
  };

  /**
   * readonly
   */
  Buffer *buffer;

private:
  Flags<BufferAllocationFlag> flags_;
  std::atomic<uint32_t> refcount_ = {0u};
  /* ml681: a deque, not a queue, purely so the retained set can be WALKED --
   * eligible vs ineligible cannot be separated without iterating, and that
   * distinction is the whole question. FIFO discipline is unchanged
   * (push_back / pop_front). */
  std::deque<QueueEntry> fifo;
public:
  uint32_t census_id_ = 0;      /* ml681: index into the per-instance stats */
private:
  dxmt::mutex mutex_;
  Rc<BufferAllocation> name_;
  uint32_t name_suballocation_ = 0;
  bool owned_by_command_list_ = false;
  bool trimmed_last_ = false;      /* ml685: trim-regret latch */
  /* MADEIRA (P4 T5): newest coherent sequence id seen by THIS instance, the
   * only domain its fifo's will_free_at values belong to (see trimEligible). */
  uint64_t last_coherent_seq_ = 0;
};

class DynamicLinearTexture {
public:
  void incRef();
  void decRef();

  Rc<TextureAllocation> allocate(uint64_t coherent_seq_id);
  void updateImmediateName(uint64_t current_seq_id, Rc<TextureAllocation> &&allocation, bool owned_by_command_list);
  void recycle(uint64_t current_seq_id, Rc<TextureAllocation> &&allocation);

  Rc<TextureAllocation>
  immediateName() {
    return name_;
  };

  void *
  mappedMemory() {
    return name_->mappedMemory;
  }

  DynamicLinearTexture(Texture *buffer, Flags<TextureAllocationFlag> flags);

  struct QueueEntry {
    Rc<TextureAllocation> allocation;
    uint64_t will_free_at;
  };
  /**
   * readonly
   */
  Texture *texture;

private:
  Flags<TextureAllocationFlag> flags_;
  std::atomic<uint32_t> refcount_ = {0u};
  std::deque<QueueEntry> fifo;
  dxmt::mutex mutex_;
  Rc<TextureAllocation> name_;
  bool owned_by_command_list_ = false;
};

} // namespace dxmt
