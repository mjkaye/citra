// Copyright 2023 Citra Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#include <atomic>
#include <condition_variable>
#include <mutex>
#include <queue>
#include "common/polyfill_thread.h"
#include "video_core/renderer_vulkan/vk_common.h"

VK_DEFINE_HANDLE(VmaAllocation)

namespace Vulkan {

class Instance;
class Swapchain;
class Scheduler;
class RenderpassCache;

struct Frame {
    u32 width{};
    u32 height{};
    u32 index{};
    VmaAllocation allocation{};
    vk::Framebuffer framebuffer{};
    vk::Image image{};
    vk::ImageView image_view{};
    vk::Semaphore render_ready{};
    vk::Fence present_done{};
    std::mutex fence_mutex{};
    vk::CommandBuffer cmdbuf{};
};

class PresentMailbox final {
    static constexpr std::size_t SWAP_CHAIN_SIZE = 6;

public:
    PresentMailbox(const Instance& instance, Swapchain& swapchain, Scheduler& scheduler,
                   RenderpassCache& renderpass_cache);
    ~PresentMailbox();

    Frame* GetRenderFrame();
    void UpdateSurface(u32 width, u32 height, vk::SurfaceKHR surface);
    void ReloadFrame(Frame* frame, u32 width, u32 height);
    void Present(Frame* frame);

private:
    void PresentThread(std::stop_token token);
    void CopyToSwapchain(Frame* frame);
    void RecreateSwapchain(u32 width, u32 height);

    struct FrameQueue {
        void Push(Frame* frame) {
            std::scoped_lock lock{mutex};
            queue.push(frame);
            cv.notify_one();
        }

        Frame* PopWait(std::stop_token token = {}) {
            std::unique_lock lock{mutex};
            if (queue.empty()) {
                Common::CondvarWait(cv, lock, token, [this] { return !queue.empty(); });
                if (token.stop_requested()) {
                    return nullptr;
                }
            }
            Frame* frame = queue.front();
            queue.pop();
            return frame;
        }

        std::queue<Frame*> queue;
        std::condition_variable_any cv;
        std::mutex mutex;
    };

private:
    const Instance& instance;
    Swapchain& swapchain;
    Scheduler& scheduler;
    RenderpassCache& renderpass_cache;
    vk::CommandPool command_pool;
    vk::Queue graphics_queue;
    std::array<Frame, SWAP_CHAIN_SIZE> swap_chain{};
    FrameQueue free_queue;
    FrameQueue present_queue;
    std::jthread present_thread;
    std::mutex swapchain_mutex;
    std::condition_variable swapchain_cv;
    bool vsync_enabled{};
};

} // namespace Vulkan
