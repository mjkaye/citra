// Copyright 2022 Citra Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once

#include <mutex>
#include <vector>
#include "common/common_types.h"
#include "video_core/renderer_vulkan/vk_common.h"

namespace Vulkan {

class Instance;
class Scheduler;
class RenderpassCache;

class Swapchain {
public:
    Swapchain(const Instance& instance, Scheduler& scheduler, u32 width, u32 height,
              RenderpassCache& renderpass_cache);
    ~Swapchain();

    /// Creates (or recreates) the swapchain with a given size.
    void Create(vk::SurfaceKHR new_surface = {}, u32 width = 1, u32 height = 1);

    /// Acquires the next image in the swapchain.
    bool AcquireNextImage();

    /// Presents the current image and move to the next one
    void Present();

    /// Returns true when the swapchain should be recreated
    [[nodiscard]] bool NeedsRecreation() const {
        return needs_recreation;
    }

    /// Notfies that the swapchain needs recreation
    void SetNeedsRecreation(bool value) noexcept {
        needs_recreation = value;
    }

    /// Returns current swapchain state
    [[nodiscard]] vk::Extent2D GetExtent() const {
        return extent;
    }

    /// Returns the swapchain surface
    [[nodiscard]] vk::SurfaceKHR GetSurface() const {
        return surface;
    }

    /// Returns the current image
    [[nodiscard]] vk::Image Image() const {
        return images[image_index];
    }

    /// Returns the swapchain format
    [[nodiscard]] vk::SurfaceFormatKHR GetSurfaceFormat() const {
        return surface_format;
    }

    /// Returns the Vulkan swapchain handle
    [[nodiscard]] vk::SwapchainKHR GetHandle() const {
        return swapchain;
    }

    [[nodiscard]] vk::Semaphore GetImageAcquiredSemaphore() const {
        return image_acquired[frame_index];
    }

    [[nodiscard]] vk::Semaphore GetPresentReadySemaphore() const {
        return present_ready[image_index];
    }

    u32 GetWidth() const {
        return width;
    }

    u32 GetHeight() const {
        return height;
    }

private:
    /// Selects the best available swapchain image format
    void FindPresentFormat();

    /// Sets the best available present mode
    void SetPresentMode();

    /// Sets the surface properties according to device capabilities
    void SetSurfaceProperties();

    /// Destroys current swapchain resources
    void Destroy();

    /// Performs creation of image views and framebuffers from the swapchain images
    void SetupImages();

    /// Creates the image acquired and present ready semaphores
    void RefreshSemaphores();

private:
    const Instance& instance;
    Scheduler& scheduler;
    RenderpassCache& renderpass_cache;
    vk::SwapchainKHR swapchain{};
    vk::SurfaceKHR surface{};
    vk::SurfaceFormatKHR surface_format;
    vk::PresentModeKHR present_mode;
    vk::Extent2D extent;
    vk::SurfaceTransformFlagBitsKHR transform;
    vk::CompositeAlphaFlagBitsKHR composite_alpha;
    std::vector<vk::Image> images;
    std::vector<vk::Semaphore> image_acquired;
    std::vector<vk::Semaphore> present_ready;
    u32 image_count = 0;
    u32 image_index = 0;
    u32 frame_index = 0;
    u32 width = 1;
    u32 height = 1;
    bool needs_recreation = true;
};

} // namespace Vulkan
