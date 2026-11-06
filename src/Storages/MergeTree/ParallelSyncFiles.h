#pragma once

#include <vector>

#include <Common/threadPoolCallbackRunner.h>
#include <IO/SharedThreadPools.h>

#include <ranges>

namespace DB
{

class WriteBufferFromFileBase;
struct MergeTreeWriterStream;

/// Calls sync() on each file in parallel using the IO thread pool.
/// fsync of multiple files inside the same part is independent and safe to run concurrently.
/// If there is only one file, it is synced inline (no thread pool overhead).
template <typename Stream>
void parallelSyncFiles(const std::vector<Stream *> & items)
{
    static_assert(std::is_same_v<Stream, WriteBufferFromFileBase> || std::is_same_v<const MergeTreeWriterStream, Stream>);
    if (items.empty())
        return;

    if (items.size() == 1 || !getIOThreadPool().isInitialized())
    {
        for (auto * item : items)
            item->sync();
        return;
    }

    /// fsync calls on different files are independent at the kernel level
    /// (subject to underlying device serialization), so we issue them in parallel.
    /// Submitting a few tasks to IOThreadPool is cheap relative to fsync latency.
    /// The last item is synced on the current thread after the remaining ones are submitted,
    /// so that `n` files occupy `n` threads instead of `n + 1` (the current thread would otherwise
    /// just block waiting for the pool).
    ThreadPoolCallbackRunnerLocal<void> runner(getIOThreadPool().get(), ThreadName::DEFAULT_THREAD_POOL);
    for (auto * item : items | std::views::take(items.size() - 1))
    {
        /// Capture `item` (a pointer) by value: `ThreadPoolCallbackRunnerLocal`
        /// forbids capturing locals by reference, since an exception while enqueuing could unwind
        /// the caller's stack before the runner's destructor waits for already-running tasks.
        runner.enqueueAndKeepTrack([item] { item->sync(); });
    }
    /// If this throws, the runner's destructor cancels the tasks not yet started and waits for the running ones.
    items.back()->sync();
    runner.waitForAllToFinishAndRethrowFirstError();
}

}
