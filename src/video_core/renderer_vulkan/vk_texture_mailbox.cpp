// Copyright 2023 Citra Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#include "common/microprofile.h"
#include "common/settings.h"
#include "common/thread.h"
#include "video_core/renderer_vulkan/vk_instance.h"
#include "video_core/renderer_vulkan/vk_renderpass_cache.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"
#include "video_core/renderer_vulkan/vk_swapchain.h"
#include "video_core/renderer_vulkan/vk_texture_mailbox.h"

#include <vk_mem_alloc.h>

MICROPROFILE_DEFINE(Vulkan_WaitPresent, "Vulkan", "Wait For Present", MP_RGB(128, 128, 128));
MICROPROFILE_DEFINE(Vulkan_PresentFrame, "Vulkan", "Present Frame", MP_RGB(64, 64, 0));

namespace Vulkan {

PresentMailbox::PresentMailbox(const Instance& instance_, Swapchain& swapchain_,
                               Scheduler& scheduler_, RenderpassCache& renderpass_cache_)
    : instance{instance_}, swapchain{swapchain_}, scheduler{scheduler_},
      renderpass_cache{renderpass_cache_}, graphics_queue{instance.GetGraphicsQueue()},
      vsync_enabled{Settings::values.use_vsync_new.GetValue()} {

    const vk::Device device = instance.GetDevice();
    const vk::CommandPoolCreateInfo pool_info = {
        .flags = vk::CommandPoolCreateFlagBits::eResetCommandBuffer |
                 vk::CommandPoolCreateFlagBits::eTransient,
        .queueFamilyIndex = instance.GetGraphicsQueueFamilyIndex(),
    };
    command_pool = device.createCommandPool(pool_info);

    const vk::CommandBufferAllocateInfo alloc_info = {
        .commandPool = command_pool,
        .level = vk::CommandBufferLevel::ePrimary,
        .commandBufferCount = SWAP_CHAIN_SIZE,
    };
    const std::vector command_buffers = device.allocateCommandBuffers(alloc_info);

    for (u32 i = 0; i < SWAP_CHAIN_SIZE; i++) {
        Frame& frame = swap_chain[i];
        frame.index = i;
        frame.cmdbuf = command_buffers[i];
        frame.render_ready = device.createSemaphore({});
        frame.present_done = device.createFence({.flags = vk::FenceCreateFlagBits::eSignaled});
        free_queue.queue.push(&frame);
    }

    present_thread = std::jthread([this](std::stop_token token) { PresentThread(token); });
}

PresentMailbox::~PresentMailbox() {
    const vk::Device device = instance.GetDevice();
    device.destroyCommandPool(command_pool);
    for (auto& frame : swap_chain) {
        device.destroyImageView(frame.image_view);
        device.destroyFramebuffer(frame.framebuffer);
        device.destroySemaphore(frame.render_ready);
        device.destroyFence(frame.present_done);
        vmaDestroyImage(instance.GetAllocator(), frame.image, frame.allocation);
    }
}

void PresentMailbox::ReloadFrame(Frame* frame, u32 width, u32 height) {
    vk::Device device = instance.GetDevice();
    if (frame->framebuffer) {
        device.destroyFramebuffer(frame->framebuffer);
    }
    if (frame->image_view) {
        device.destroyImageView(frame->image_view);
    }
    if (frame->image) {
        vmaDestroyImage(instance.GetAllocator(), frame->image, frame->allocation);
    }

    const vk::Format format = swapchain.GetSurfaceFormat().format;
    const vk::ImageCreateInfo image_info = {
        .imageType = vk::ImageType::e2D,
        .format = format,
        .extent = {width, height, 1},
        .mipLevels = 1,
        .arrayLayers = 1,
        .samples = vk::SampleCountFlagBits::e1,
        .usage = vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eTransferSrc,
    };

    const VmaAllocationCreateInfo alloc_info = {
        .flags = VMA_ALLOCATION_CREATE_WITHIN_BUDGET_BIT,
        .usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
        .requiredFlags = 0,
        .preferredFlags = 0,
        .pool = VK_NULL_HANDLE,
        .pUserData = nullptr,
    };

    VkImage unsafe_image{};
    VkImageCreateInfo unsafe_image_info = static_cast<VkImageCreateInfo>(image_info);

    VkResult result = vmaCreateImage(instance.GetAllocator(), &unsafe_image_info, &alloc_info,
                                     &unsafe_image, &frame->allocation, nullptr);
    if (result != VK_SUCCESS) [[unlikely]] {
        LOG_CRITICAL(Render_Vulkan, "Failed allocating texture with error {}", result);
        UNREACHABLE();
    }
    frame->image = vk::Image{unsafe_image};

    const vk::ImageViewCreateInfo view_info = {
        .image = frame->image,
        .viewType = vk::ImageViewType::e2D,
        .format = format,
        .subresourceRange{
            .aspectMask = vk::ImageAspectFlagBits::eColor,
            .baseMipLevel = 0,
            .levelCount = 1,
            .baseArrayLayer = 0,
            .layerCount = 1,
        },
    };
    frame->image_view = device.createImageView(view_info);

    const vk::FramebufferCreateInfo framebuffer_info = {
        .renderPass = renderpass_cache.GetPresentRenderpass(),
        .attachmentCount = 1,
        .pAttachments = &frame->image_view,
        .width = width,
        .height = height,
        .layers = 1,
    };
    frame->framebuffer = instance.GetDevice().createFramebuffer(framebuffer_info);

    frame->width = width;
    frame->height = height;
}

Frame* PresentMailbox::GetRenderFrame() {
    MICROPROFILE_SCOPE(Vulkan_WaitPresent);

    Frame* frame = free_queue.PopWait();

    vk::Device device = instance.GetDevice();
    vk::Result result{};

    const auto Wait = [&]() {
        std::scoped_lock lock{frame->fence_mutex};
        result = device.waitForFences(frame->present_done, false, std::numeric_limits<u64>::max());
        return result;
    };

    while (Wait() != vk::Result::eSuccess) {
        // Retry if the waiting time out
        if (result == vk::Result::eTimeout) {
            continue;
        }

        // eErrorInitializationFailed occurs on Mali GPU drivers due to them
        // using the ppoll() syscall which isn't correctly restarted after a signal,
        // we need to manually retry waiting in that case
        if (result == vk::Result::eErrorInitializationFailed) {
            continue;
        }
    }

    device.resetFences(frame->present_done);
    return frame;
}

void PresentMailbox::UpdateSurface(u32 width, u32 height, vk::SurfaceKHR surface) {
    std::scoped_lock lock{swapchain_mutex};
    swapchain.Create(width, height, surface);
    swapchain_cv.notify_one();
}

void PresentMailbox::Present(Frame* frame) {
    present_queue.Push(frame);
}

void PresentMailbox::PresentThread(std::stop_token token) {
    Common::SetCurrentThreadName("VulkanPresent");
    do {
        Frame* frame = present_queue.PopWait(token);
        if (token.stop_requested()) {
            continue;
        }
        CopyToSwapchain(frame);
        free_queue.Push(frame);
    } while (!token.stop_requested());
}

void PresentMailbox::CopyToSwapchain(Frame* frame) {
    MICROPROFILE_SCOPE(Vulkan_PresentFrame);

#if ANDROID
    // On Android swapchain invalidations are always due to surface changes.
    // These are processed on the main thread so wait for it to recreate
    // the swapchain for us.
    std::unique_lock lock{swapchain_mutex};
    swapchain_cv.wait(lock, [this]() { return !swapchain.NeedsRecreation(); });
#endif

    // Check if the vsync setting was changed
    const bool use_vsync = Settings::values.use_vsync_new.GetValue();
    if (vsync_enabled != use_vsync) [[unlikely]] {
        vsync_enabled = use_vsync;
        RecreateSwapchain(frame->width, frame->height);
    }

    while (!swapchain.AcquireNextImage()) {
#if ANDROID
        swapchain_cv.wait(lock, [this]() { return !swapchain.NeedsRecreation(); });
#else
        RecreateSwapchain(frame->width, frame->height);
#endif
    }

    const vk::Image swapchain_image = swapchain.Image();

    const vk::CommandBufferBeginInfo begin_info = {
        .flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit,
    };
    const vk::CommandBuffer cmdbuf = frame->cmdbuf;
    cmdbuf.begin(begin_info);

    const auto [width, height] = swapchain.GetExtent();
    const u32 copy_width = std::min(width, frame->width);
    const u32 copy_height = std::min(height, frame->height);

    const vk::ImageCopy image_copy = {
        .srcSubresource{
            .aspectMask = vk::ImageAspectFlagBits::eColor,
            .mipLevel = 0,
            .baseArrayLayer = 0,
            .layerCount = 1,
        },
        .srcOffset = {0, 0, 0},
        .dstSubresource{
            .aspectMask = vk::ImageAspectFlagBits::eColor,
            .mipLevel = 0,
            .baseArrayLayer = 0,
            .layerCount = 1,
        },
        .dstOffset = {0, 0, 0},
        .extent = {copy_width, copy_height, 1},
    };

    const std::array pre_barriers{
        vk::ImageMemoryBarrier{
            .srcAccessMask = vk::AccessFlagBits::eNone,
            .dstAccessMask = vk::AccessFlagBits::eTransferWrite,
            .oldLayout = vk::ImageLayout::eUndefined,
            .newLayout = vk::ImageLayout::eTransferDstOptimal,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .image = swapchain_image,
            .subresourceRange{
                .aspectMask = vk::ImageAspectFlagBits::eColor,
                .baseMipLevel = 0,
                .levelCount = 1,
                .baseArrayLayer = 0,
                .layerCount = VK_REMAINING_ARRAY_LAYERS,
            },
        },
        vk::ImageMemoryBarrier{
            .srcAccessMask = vk::AccessFlagBits::eColorAttachmentWrite,
            .dstAccessMask = vk::AccessFlagBits::eTransferRead,
            .oldLayout = vk::ImageLayout::eTransferSrcOptimal,
            .newLayout = vk::ImageLayout::eTransferSrcOptimal,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .image = frame->image,
            .subresourceRange{
                .aspectMask = vk::ImageAspectFlagBits::eColor,
                .baseMipLevel = 0,
                .levelCount = 1,
                .baseArrayLayer = 0,
                .layerCount = VK_REMAINING_ARRAY_LAYERS,
            },
        },
    };
    const vk::ImageMemoryBarrier post_barrier{
        .srcAccessMask = vk::AccessFlagBits::eTransferWrite,
        .dstAccessMask = vk::AccessFlagBits::eMemoryRead,
        .oldLayout = vk::ImageLayout::eTransferDstOptimal,
        .newLayout = vk::ImageLayout::ePresentSrcKHR,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .image = swapchain_image,
        .subresourceRange{
            .aspectMask = vk::ImageAspectFlagBits::eColor,
            .baseMipLevel = 0,
            .levelCount = 1,
            .baseArrayLayer = 0,
            .layerCount = VK_REMAINING_ARRAY_LAYERS,
        },
    };

    cmdbuf.pipelineBarrier(vk::PipelineStageFlagBits::eColorAttachmentOutput,
                           vk::PipelineStageFlagBits::eTransfer, vk::DependencyFlagBits::eByRegion,
                           {}, {}, pre_barriers);

    cmdbuf.copyImage(frame->image, vk::ImageLayout::eTransferSrcOptimal, swapchain_image,
                     vk::ImageLayout::eTransferDstOptimal, image_copy);

    cmdbuf.pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands,
                           vk::PipelineStageFlagBits::eAllCommands,
                           vk::DependencyFlagBits::eByRegion, {}, {}, post_barrier);

    cmdbuf.end();

    static constexpr std::array<vk::PipelineStageFlags, 2> wait_stage_masks = {
        vk::PipelineStageFlagBits::eColorAttachmentOutput,
        vk::PipelineStageFlagBits::eAllGraphics,
    };

    const vk::Semaphore present_ready = swapchain.GetPresentReadySemaphore();
    const vk::Semaphore image_acquired = swapchain.GetImageAcquiredSemaphore();
    const std::array wait_semaphores = {image_acquired, frame->render_ready};

    vk::SubmitInfo submit_info = {
        .waitSemaphoreCount = static_cast<u32>(wait_semaphores.size()),
        .pWaitSemaphores = wait_semaphores.data(),
        .pWaitDstStageMask = wait_stage_masks.data(),
        .commandBufferCount = 1u,
        .pCommandBuffers = &cmdbuf,
        .signalSemaphoreCount = 1,
        .pSignalSemaphores = &present_ready,
    };

    try {
        std::scoped_lock lock{scheduler.QueueMutex(), frame->fence_mutex};
        graphics_queue.submit(submit_info, frame->present_done);
    } catch (vk::DeviceLostError& err) {
        LOG_CRITICAL(Render_Vulkan, "Device lost during present submit: {}", err.what());
        UNREACHABLE();
    }

    swapchain.Present();
}

void PresentMailbox::RecreateSwapchain(u32 width, u32 height) {
    std::scoped_lock lock{scheduler.QueueMutex()};
    graphics_queue.waitIdle();
    swapchain.Create(width, height);
}

} // namespace Vulkan
