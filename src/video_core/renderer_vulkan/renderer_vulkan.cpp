// Copyright 2022 Citra Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/gtc/matrix_transform.hpp>
#include "common/assert.h"
#include "common/logging/log.h"
#include "common/microprofile.h"
#include "common/settings.h"
#include "core/core.h"
#include "core/frontend/framebuffer_layout.h"
#include "core/hw/gpu.h"
#include "core/hw/hw.h"
#include "core/hw/lcd.h"
#include "core/tracer/recorder.h"
#include "video_core/debug_utils/debug_utils.h"
#include "video_core/renderer_vulkan/renderer_vulkan.h"
#include "video_core/renderer_vulkan/vk_platform.h"
#include "video_core/renderer_vulkan/vk_shader_util.h"
#include "video_core/renderer_vulkan/vk_texture_mailbox.h"
#include "video_core/video_core.h"

#include "video_core/host_shaders/vulkan_present_anaglyph_frag_spv.h"
#include "video_core/host_shaders/vulkan_present_frag_spv.h"
#include "video_core/host_shaders/vulkan_present_interlaced_frag_spv.h"
#include "video_core/host_shaders/vulkan_present_vert_spv.h"

#include <vk_mem_alloc.h>

MICROPROFILE_DEFINE(Vulkan_RenderFrame, "Vulkan", "Render Frame", MP_RGB(128, 128, 64));

namespace Vulkan {

/**
 * Vertex structure that the drawn screen rectangles are composed of.
 */
struct ScreenRectVertex {
    ScreenRectVertex() = default;
    ScreenRectVertex(float x, float y, float u, float v)
        : position{Common::MakeVec(x, y)}, tex_coord{Common::MakeVec(u, v)} {}

    Common::Vec2f position;
    Common::Vec2f tex_coord;
};

constexpr u32 VERTEX_BUFFER_SIZE = sizeof(ScreenRectVertex) * 8192;

constexpr std::array<f32, 4 * 4> MakeOrthographicMatrix(f32 width, f32 height) {
    // clang-format off
    return { 2.f / width, 0.f,          0.f, 0.f,
             0.f,         2.f / height, 0.f, 0.f,
             0.f,         0.f,          1.f, 0.f,
            -1.f,        -1.f,          0.f, 1.f};
    // clang-format on
}

namespace {
std::string GetReadableVersion(u32 version) {
    return fmt::format("{}.{}.{}", VK_VERSION_MAJOR(version), VK_VERSION_MINOR(version),
                       VK_VERSION_PATCH(version));
}

std::string GetDriverVersion(const Instance& instance) {
    // Extracted from
    // https://github.com/SaschaWillems/vulkan.gpuinfo.org/blob/5dddea46ea1120b0df14eef8f15ff8e318e35462/functions.php#L308-L314
    const u32 version = instance.GetDriverVersion();
    if (instance.GetDriverID() == vk::DriverId::eNvidiaProprietary) {
        const u32 major = (version >> 22) & 0x3ff;
        const u32 minor = (version >> 14) & 0x0ff;
        const u32 secondary = (version >> 6) & 0x0ff;
        const u32 tertiary = version & 0x003f;
        return fmt::format("{}.{}.{}.{}", major, minor, secondary, tertiary);
    }
    if (instance.GetDriverID() == vk::DriverId::eIntelProprietaryWindows) {
        const u32 major = version >> 14;
        const u32 minor = version & 0x3fff;
        return fmt::format("{}.{}", major, minor);
    }
    return GetReadableVersion(version);
}

std::string BuildCommaSeparatedExtensions(std::vector<std::string> available_extensions) {
    std::sort(available_extensions.begin(), available_extensions.end());

    static constexpr std::size_t AverageExtensionSize = 64;
    std::string separated_extensions;
    separated_extensions.reserve(available_extensions.size() * AverageExtensionSize);

    const auto end = std::end(available_extensions);
    for (auto extension = std::begin(available_extensions); extension != end; ++extension) {
        if (const bool is_last = extension + 1 == end; is_last) {
            separated_extensions += *extension;
        } else {
            separated_extensions += fmt::format("{},", *extension);
        }
    }
    return separated_extensions;
}

} // Anonymous namespace

RendererVulkan::RendererVulkan(Core::System& system_, Frontend::EmuWindow& window,
                               Frontend::EmuWindow* secondary_window)
    : RendererBase{window, secondary_window}, system{system_}, memory{system.Memory()},
      telemetry_session{system.TelemetrySession()},
      instance{window, Settings::values.physical_device.GetValue()}, scheduler{instance,
                                                                               renderpass_cache},
      renderpass_cache{instance, scheduler}, desc_manager{instance, scheduler},
      runtime{instance, scheduler, renderpass_cache, desc_manager},
      swapchain{instance, scheduler, render_window.GetFramebufferLayout().width,
                render_window.GetFramebufferLayout().height, renderpass_cache},
      vertex_buffer{instance, scheduler, vk::BufferUsageFlagBits::eVertexBuffer,
                    VERTEX_BUFFER_SIZE},
      rasterizer{
          memory,  system.CustomTexManager(), render_window, instance, scheduler, desc_manager,
          runtime, renderpass_cache} {
    Report();
    CompileShaders();
    BuildLayouts();
    BuildPipelines();
    mailbox = std::make_unique<PresentMailbox>(instance, swapchain, scheduler, renderpass_cache);
}

RendererVulkan::~RendererVulkan() {
    vk::Device device = instance.GetDevice();
    device.waitIdle();

    device.destroyPipelineLayout(present_pipeline_layout);
    device.destroyShaderModule(present_vertex_shader);
    device.destroyDescriptorSetLayout(present_descriptor_layout);
    device.destroyDescriptorUpdateTemplate(present_update_template);

    for (u32 i = 0; i < PRESENT_PIPELINES; i++) {
        device.destroyPipeline(present_pipelines[i]);
        device.destroyShaderModule(present_shaders[i]);
    }

    for (auto& sampler : present_samplers) {
        device.destroySampler(sampler);
    }

    for (auto& info : screen_infos) {
        device.destroyImageView(info.texture.image_view);
        vmaDestroyImage(instance.GetAllocator(), info.texture.image, info.texture.allocation);
    }

    mailbox.reset();
}

void RendererVulkan::Sync() {
    rasterizer.SyncEntireState();
}

void RendererVulkan::PrepareRendertarget() {
    for (u32 i = 0; i < 3; i++) {
        const u32 fb_id = i == 2 ? 1 : 0;
        const auto& framebuffer = GPU::g_regs.framebuffer_config[fb_id];

        // Main LCD (0): 0x1ED02204, Sub LCD (1): 0x1ED02A04
        u32 lcd_color_addr =
            (fb_id == 0) ? LCD_REG_INDEX(color_fill_top) : LCD_REG_INDEX(color_fill_bottom);
        lcd_color_addr = HW::VADDR_LCD + 4 * lcd_color_addr;
        LCD::Regs::ColorFill color_fill{0};
        LCD::Read(color_fill.raw, lcd_color_addr);

        if (color_fill.is_enabled) {
            LoadColorToActiveVkTexture(color_fill.color_r, color_fill.color_g, color_fill.color_b,
                                       screen_infos[i].texture);
        } else {
            TextureInfo& texture = screen_infos[i].texture;
            if (texture.width != framebuffer.width || texture.height != framebuffer.height ||
                texture.format != framebuffer.color_format) {

                // Reallocate texture if the framebuffer size has changed.
                // This is expected to not happen very often and hence should not be a
                // performance problem.
                ConfigureFramebufferTexture(texture, framebuffer);
            }

            LoadFBToScreenInfo(framebuffer, screen_infos[i], i == 1);

            // Resize the texture in case the framebuffer size has changed
            texture.width = framebuffer.width;
            texture.height = framebuffer.height;
        }
    }
}

void RendererVulkan::RenderToMailbox(const Layout::FramebufferLayout& layout,
                                     std::unique_ptr<PresentMailbox>& mailbox, bool flipped) {
    Frame* frame = mailbox->GetRenderFrame();
    MICROPROFILE_SCOPE(Vulkan_RenderFrame);

    const auto [width, height] = swapchain.GetExtent();
    if (width != frame->width || height != frame->height) {
        mailbox->ReloadFrame(frame, width, height);
    }

    scheduler.Record([layout](vk::CommandBuffer cmdbuf) {
        const vk::Viewport viewport = {
            .x = 0.0f,
            .y = 0.0f,
            .width = static_cast<float>(layout.width),
            .height = static_cast<float>(layout.height),
            .minDepth = 0.0f,
            .maxDepth = 1.0f,
        };

        const vk::Rect2D scissor = {
            .offset = {0, 0},
            .extent = {layout.width, layout.height},
        };

        cmdbuf.setViewport(0, viewport);
        cmdbuf.setScissor(0, scissor);
    });

    DrawScreens(frame, layout, flipped);

    scheduler.Flush(frame->render_ready);
    scheduler.Record([&mailbox, frame](vk::CommandBuffer) { mailbox->Present(frame); });
    scheduler.DispatchWork();
}

void RendererVulkan::BeginRendering(Frame* frame) {
    for (std::size_t i = 0; i < screen_infos.size(); i++) {
        const auto& info = screen_infos[i];
        present_textures[i] = vk::DescriptorImageInfo{
            .sampler = present_samplers[current_sampler],
            .imageView = info.image_view,
            .imageLayout = vk::ImageLayout::eGeneral,
        };
    }

    vk::DescriptorSet set = desc_manager.AllocateSet(present_descriptor_layout);
    instance.GetDevice().updateDescriptorSetWithTemplate(set, present_update_template,
                                                         present_textures[0]);

    renderpass_cache.EndRendering();
    scheduler.Record([this, framebuffer = frame->framebuffer, width = frame->width,
                      height = frame->height, set,
                      pipeline_index = current_pipeline](vk::CommandBuffer cmdbuf) {
        const vk::ClearValue clear{.color = clear_color};
        const vk::RenderPassBeginInfo renderpass_begin_info = {
            .renderPass = renderpass_cache.GetPresentRenderpass(),
            .framebuffer = framebuffer,
            .renderArea =
                vk::Rect2D{
                    .offset = {0, 0},
                    .extent = {width, height},
                },
            .clearValueCount = 1,
            .pClearValues = &clear,
        };

        cmdbuf.beginRenderPass(renderpass_begin_info, vk::SubpassContents::eInline);
        cmdbuf.bindPipeline(vk::PipelineBindPoint::eGraphics, present_pipelines[pipeline_index]);
        cmdbuf.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, present_pipeline_layout, 0, set,
                                  {});
    });
}

void RendererVulkan::LoadFBToScreenInfo(const GPU::Regs::FramebufferConfig& framebuffer,
                                        ScreenInfo& screen_info, bool right_eye) {

    if (framebuffer.address_right1 == 0 || framebuffer.address_right2 == 0)
        right_eye = false;

    const PAddr framebuffer_addr =
        framebuffer.active_fb == 0
            ? (!right_eye ? framebuffer.address_left1 : framebuffer.address_right1)
            : (!right_eye ? framebuffer.address_left2 : framebuffer.address_right2);

    LOG_TRACE(Render_Vulkan, "0x{:08x} bytes from 0x{:08x}({}x{}), fmt {:x}",
              framebuffer.stride * framebuffer.height, framebuffer_addr, framebuffer.width.Value(),
              framebuffer.height.Value(), framebuffer.format);

    int bpp = GPU::Regs::BytesPerPixel(framebuffer.color_format);
    std::size_t pixel_stride = framebuffer.stride / bpp;

    ASSERT(pixel_stride * bpp == framebuffer.stride);
    ASSERT(pixel_stride % 4 == 0);

    if (!rasterizer.AccelerateDisplay(framebuffer, framebuffer_addr, static_cast<u32>(pixel_stride),
                                      screen_info)) {
        // Reset the screen info's display texture to its own permanent texture
        screen_info.image_view = screen_info.texture.image_view;
        screen_info.texcoords = {0.f, 0.f, 1.f, 1.f};

        ASSERT(false);
    }
}

void RendererVulkan::CompileShaders() {
    vk::Device device = instance.GetDevice();
    present_vertex_shader = CompileSPV(VULKAN_PRESENT_VERT_SPV, device);
    present_shaders[0] = CompileSPV(VULKAN_PRESENT_FRAG_SPV, device);
    present_shaders[1] = CompileSPV(VULKAN_PRESENT_ANAGLYPH_FRAG_SPV, device);
    present_shaders[2] = CompileSPV(VULKAN_PRESENT_INTERLACED_FRAG_SPV, device);

    auto properties = instance.GetPhysicalDevice().getProperties();
    for (std::size_t i = 0; i < present_samplers.size(); i++) {
        const vk::Filter filter_mode = i == 0 ? vk::Filter::eLinear : vk::Filter::eNearest;
        const vk::SamplerCreateInfo sampler_info = {
            .magFilter = filter_mode,
            .minFilter = filter_mode,
            .mipmapMode = vk::SamplerMipmapMode::eLinear,
            .addressModeU = vk::SamplerAddressMode::eClampToEdge,
            .addressModeV = vk::SamplerAddressMode::eClampToEdge,
            .anisotropyEnable = instance.IsAnisotropicFilteringSupported(),
            .maxAnisotropy = properties.limits.maxSamplerAnisotropy,
            .compareEnable = false,
            .compareOp = vk::CompareOp::eAlways,
            .borderColor = vk::BorderColor::eIntOpaqueBlack,
            .unnormalizedCoordinates = false,
        };

        present_samplers[i] = device.createSampler(sampler_info);
    }
}

void RendererVulkan::BuildLayouts() {
    const vk::DescriptorSetLayoutBinding present_layout_binding = {
        .binding = 0,
        .descriptorType = vk::DescriptorType::eCombinedImageSampler,
        .descriptorCount = 3,
        .stageFlags = vk::ShaderStageFlagBits::eFragment,
    };

    const vk::DescriptorSetLayoutCreateInfo present_layout_info = {
        .bindingCount = 1,
        .pBindings = &present_layout_binding,
    };

    const vk::Device device = instance.GetDevice();
    present_descriptor_layout = device.createDescriptorSetLayout(present_layout_info);

    const vk::DescriptorUpdateTemplateEntry update_template_entry = {
        .dstBinding = 0,
        .dstArrayElement = 0,
        .descriptorCount = 3,
        .descriptorType = vk::DescriptorType::eCombinedImageSampler,
        .offset = 0,
        .stride = sizeof(vk::DescriptorImageInfo),
    };

    const vk::DescriptorUpdateTemplateCreateInfo template_info = {
        .descriptorUpdateEntryCount = 1,
        .pDescriptorUpdateEntries = &update_template_entry,
        .templateType = vk::DescriptorUpdateTemplateType::eDescriptorSet,
        .descriptorSetLayout = present_descriptor_layout,
    };
    present_update_template = device.createDescriptorUpdateTemplate(template_info);

    const vk::PushConstantRange push_range = {
        .stageFlags = vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment,
        .offset = 0,
        .size = sizeof(PresentUniformData),
    };

    const vk::PipelineLayoutCreateInfo layout_info = {
        .setLayoutCount = 1,
        .pSetLayouts = &present_descriptor_layout,
        .pushConstantRangeCount = 1,
        .pPushConstantRanges = &push_range,
    };
    present_pipeline_layout = device.createPipelineLayout(layout_info);
}

void RendererVulkan::BuildPipelines() {
    const vk::VertexInputBindingDescription binding = {
        .binding = 0,
        .stride = sizeof(ScreenRectVertex),
        .inputRate = vk::VertexInputRate::eVertex,
    };

    const std::array attributes = {
        vk::VertexInputAttributeDescription{
            .location = 0,
            .binding = 0,
            .format = vk::Format::eR32G32Sfloat,
            .offset = offsetof(ScreenRectVertex, position),
        },
        vk::VertexInputAttributeDescription{
            .location = 1,
            .binding = 0,
            .format = vk::Format::eR32G32Sfloat,
            .offset = offsetof(ScreenRectVertex, tex_coord),
        },
    };

    const vk::PipelineVertexInputStateCreateInfo vertex_input_info = {
        .vertexBindingDescriptionCount = 1,
        .pVertexBindingDescriptions = &binding,
        .vertexAttributeDescriptionCount = static_cast<u32>(attributes.size()),
        .pVertexAttributeDescriptions = attributes.data(),
    };

    const vk::PipelineInputAssemblyStateCreateInfo input_assembly = {
        .topology = vk::PrimitiveTopology::eTriangleStrip,
        .primitiveRestartEnable = false,
    };

    const vk::PipelineRasterizationStateCreateInfo raster_state = {
        .depthClampEnable = false,
        .rasterizerDiscardEnable = false,
        .cullMode = vk::CullModeFlagBits::eNone,
        .frontFace = vk::FrontFace::eClockwise,
        .depthBiasEnable = false,
        .lineWidth = 1.0f,
    };

    const vk::PipelineMultisampleStateCreateInfo multisampling = {
        .rasterizationSamples = vk::SampleCountFlagBits::e1,
        .sampleShadingEnable = false,
    };

    const vk::PipelineColorBlendAttachmentState colorblend_attachment = {
        .blendEnable = false,
        .colorWriteMask = vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG |
                          vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA,
    };

    const vk::PipelineColorBlendStateCreateInfo color_blending = {
        .logicOpEnable = false,
        .attachmentCount = 1,
        .pAttachments = &colorblend_attachment,
        .blendConstants = std::array{1.0f, 1.0f, 1.0f, 1.0f},
    };

    const vk::Viewport placeholder_viewport = vk::Viewport{0.0f, 0.0f, 1.0f, 1.0f, 0.0f, 1.0f};
    const vk::Rect2D placeholder_scissor = vk::Rect2D{{0, 0}, {1, 1}};
    const vk::PipelineViewportStateCreateInfo viewport_info = {
        .viewportCount = 1,
        .pViewports = &placeholder_viewport,
        .scissorCount = 1,
        .pScissors = &placeholder_scissor,
    };

    const std::array dynamic_states = {
        vk::DynamicState::eViewport,
        vk::DynamicState::eScissor,
    };

    const vk::PipelineDynamicStateCreateInfo dynamic_info = {
        .dynamicStateCount = static_cast<u32>(dynamic_states.size()),
        .pDynamicStates = dynamic_states.data(),
    };

    const vk::PipelineDepthStencilStateCreateInfo depth_info = {
        .depthTestEnable = false,
        .depthWriteEnable = false,
        .depthCompareOp = vk::CompareOp::eAlways,
        .depthBoundsTestEnable = false,
        .stencilTestEnable = false,
    };

    for (u32 i = 0; i < PRESENT_PIPELINES; i++) {
        const std::array shader_stages = {
            vk::PipelineShaderStageCreateInfo{
                .stage = vk::ShaderStageFlagBits::eVertex,
                .module = present_vertex_shader,
                .pName = "main",
            },
            vk::PipelineShaderStageCreateInfo{
                .stage = vk::ShaderStageFlagBits::eFragment,
                .module = present_shaders[i],
                .pName = "main",
            },
        };

        const vk::GraphicsPipelineCreateInfo pipeline_info = {
            .stageCount = static_cast<u32>(shader_stages.size()),
            .pStages = shader_stages.data(),
            .pVertexInputState = &vertex_input_info,
            .pInputAssemblyState = &input_assembly,
            .pViewportState = &viewport_info,
            .pRasterizationState = &raster_state,
            .pMultisampleState = &multisampling,
            .pDepthStencilState = &depth_info,
            .pColorBlendState = &color_blending,
            .pDynamicState = &dynamic_info,
            .layout = present_pipeline_layout,
            .renderPass = renderpass_cache.GetPresentRenderpass(),
        };

        vk::Device device = instance.GetDevice();
        if (const auto result = device.createGraphicsPipeline({}, pipeline_info);
            result.result == vk::Result::eSuccess) {
            present_pipelines[i] = result.value;
        } else {
            LOG_CRITICAL(Render_Vulkan, "Unable to build present pipelines");
            UNREACHABLE();
        }
    }
}

void RendererVulkan::ConfigureFramebufferTexture(TextureInfo& texture,
                                                 const GPU::Regs::FramebufferConfig& framebuffer) {
    vk::Device device = instance.GetDevice();
    if (texture.image_view) {
        device.destroyImageView(texture.image_view);
    }
    if (texture.image) {
        vmaDestroyImage(instance.GetAllocator(), texture.image, texture.allocation);
    }

    const VideoCore::PixelFormat pixel_format =
        VideoCore::PixelFormatFromGPUPixelFormat(framebuffer.color_format);
    const vk::Format format = instance.GetTraits(pixel_format).native;
    const vk::ImageCreateInfo image_info = {
        .imageType = vk::ImageType::e2D,
        .format = format,
        .extent = {framebuffer.width, framebuffer.height, 1},
        .mipLevels = 1,
        .arrayLayers = 1,
        .samples = vk::SampleCountFlagBits::e1,
        .usage = vk::ImageUsageFlagBits::eSampled,
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
                                     &unsafe_image, &texture.allocation, nullptr);
    if (result != VK_SUCCESS) [[unlikely]] {
        LOG_CRITICAL(Render_Vulkan, "Failed allocating texture with error {}", result);
        UNREACHABLE();
    }
    texture.image = vk::Image{unsafe_image};

    const vk::ImageViewCreateInfo view_info = {
        .image = texture.image,
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
    texture.image_view = device.createImageView(view_info);

    texture.width = framebuffer.width;
    texture.height = framebuffer.height;
    texture.format = framebuffer.color_format;
}

void RendererVulkan::LoadColorToActiveVkTexture(u8 color_r, u8 color_g, u8 color_b,
                                                const TextureInfo& texture) {
    const vk::ClearColorValue clear_color = {
        .float32 =
            std::array{
                color_r / 255.0f,
                color_g / 255.0f,
                color_b / 255.0f,
                1.0f,
            },
    };

    renderpass_cache.EndRendering();
    scheduler.Record([image = texture.image, clear_color](vk::CommandBuffer cmdbuf) {
        const vk::ImageSubresourceRange range = {
            .aspectMask = vk::ImageAspectFlagBits::eColor,
            .baseMipLevel = 0,
            .levelCount = VK_REMAINING_MIP_LEVELS,
            .baseArrayLayer = 0,
            .layerCount = VK_REMAINING_ARRAY_LAYERS,
        };

        const vk::ImageMemoryBarrier pre_barrier = {
            .srcAccessMask = vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eTransferRead,
            .dstAccessMask = vk::AccessFlagBits::eTransferWrite,
            .oldLayout = vk::ImageLayout::eGeneral,
            .newLayout = vk::ImageLayout::eTransferDstOptimal,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .image = image,
            .subresourceRange = range,
        };

        const vk::ImageMemoryBarrier post_barrier = {
            .srcAccessMask = vk::AccessFlagBits::eTransferWrite,
            .dstAccessMask = vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eTransferRead,
            .oldLayout = vk::ImageLayout::eTransferDstOptimal,
            .newLayout = vk::ImageLayout::eGeneral,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .image = image,
            .subresourceRange = range,
        };

        cmdbuf.pipelineBarrier(vk::PipelineStageFlagBits::eFragmentShader,
                               vk::PipelineStageFlagBits::eTransfer,
                               vk::DependencyFlagBits::eByRegion, {}, {}, pre_barrier);

        cmdbuf.clearColorImage(image, vk::ImageLayout::eTransferDstOptimal, clear_color, range);

        cmdbuf.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
                               vk::PipelineStageFlagBits::eFragmentShader,
                               vk::DependencyFlagBits::eByRegion, {}, {}, post_barrier);
    });
}

void RendererVulkan::ReloadSampler() {
    current_sampler = !Settings::values.filter_mode.GetValue();
}

void RendererVulkan::ReloadPipeline() {
    const Settings::StereoRenderOption render_3d = Settings::values.render_3d.GetValue();
    switch (render_3d) {
    case Settings::StereoRenderOption::Anaglyph:
        current_pipeline = 1;
        break;
    case Settings::StereoRenderOption::Interlaced:
    case Settings::StereoRenderOption::ReverseInterlaced:
        current_pipeline = 2;
        draw_info.reverse_interlaced = render_3d == Settings::StereoRenderOption::ReverseInterlaced;
        break;
    default:
        current_pipeline = 0;
        break;
    }
}

void RendererVulkan::DrawSingleScreenRotated(u32 screen_id, float x, float y, float w, float h) {
    auto& screen_info = screen_infos[screen_id];
    const auto& texcoords = screen_info.texcoords;

    u32 size = sizeof(ScreenRectVertex) * 4;
    auto [ptr, offset, invalidate] = vertex_buffer.Map(size, 16);

    const std::array vertices = {
        ScreenRectVertex{x, y, texcoords.bottom, texcoords.left},
        ScreenRectVertex{x + w, y, texcoords.bottom, texcoords.right},
        ScreenRectVertex{x, y + h, texcoords.top, texcoords.left},
        ScreenRectVertex{x + w, y + h, texcoords.top, texcoords.right},
    };

    std::memcpy(ptr, vertices.data(), size);
    vertex_buffer.Commit(size);

    // As this is the "DrawSingleScreenRotated" function, the output resolution dimensions have been
    // swapped. If a non-rotated draw-screen function were to be added for book-mode games, those
    // should probably be set to the standard (w, h, 1.0 / w, 1.0 / h) ordering.
    const u16 scale_factor = VideoCore::GetResolutionScaleFactor();
    const float width = static_cast<float>(screen_info.texture.width);
    const float height = static_cast<float>(screen_info.texture.height);

    draw_info.i_resolution =
        Common::Vec4f{width * scale_factor, height * scale_factor, 1.0f / (width * scale_factor),
                      1.0f / (height * scale_factor)};
    draw_info.o_resolution = Common::Vec4f{h, w, 1.0f / h, 1.0f / w};
    draw_info.screen_id_l = screen_id;

    scheduler.Record([this, offset = offset, info = draw_info](vk::CommandBuffer cmdbuf) {
        cmdbuf.pushConstants(present_pipeline_layout,
                             vk::ShaderStageFlagBits::eFragment | vk::ShaderStageFlagBits::eVertex,
                             0, sizeof(info), &info);

        cmdbuf.bindVertexBuffers(0, vertex_buffer.Handle(), {0});
        cmdbuf.draw(4, 1, offset / sizeof(ScreenRectVertex), 0);
    });
}

void RendererVulkan::DrawSingleScreen(u32 screen_id, float x, float y, float w, float h) {
    auto& screen_info = screen_infos[screen_id];
    const auto& texcoords = screen_info.texcoords;

    u32 size = sizeof(ScreenRectVertex) * 4;
    auto [ptr, offset, invalidate] = vertex_buffer.Map(size, 16);

    const std::array vertices = {
        ScreenRectVertex{x, y, texcoords.bottom, texcoords.right},
        ScreenRectVertex{x + w, y, texcoords.top, texcoords.right},
        ScreenRectVertex{x, y + h, texcoords.bottom, texcoords.left},
        ScreenRectVertex{x + w, y + h, texcoords.top, texcoords.left},
    };

    std::memcpy(ptr, vertices.data(), size);
    vertex_buffer.Commit(size);

    const u16 scale_factor = VideoCore::GetResolutionScaleFactor();
    const float width = static_cast<float>(screen_info.texture.width);
    const float height = static_cast<float>(screen_info.texture.height);

    draw_info.i_resolution =
        Common::Vec4f{width * scale_factor, height * scale_factor, 1.0f / (width * scale_factor),
                      1.0f / (height * scale_factor)};
    draw_info.o_resolution = Common::Vec4f{w, h, 1.0f / h, 1.0f / w};
    draw_info.screen_id_l = screen_id;

    scheduler.Record([this, offset = offset, info = draw_info](vk::CommandBuffer cmdbuf) {
        cmdbuf.pushConstants(present_pipeline_layout,
                             vk::ShaderStageFlagBits::eFragment | vk::ShaderStageFlagBits::eVertex,
                             0, sizeof(info), &info);

        cmdbuf.bindVertexBuffers(0, vertex_buffer.Handle(), {0});
        cmdbuf.draw(4, 1, offset / sizeof(ScreenRectVertex), 0);
    });
}

void RendererVulkan::DrawSingleScreenStereoRotated(u32 screen_id_l, u32 screen_id_r, float x,
                                                   float y, float w, float h) {
    const ScreenInfo& screen_info_l = screen_infos[screen_id_l];
    const auto& texcoords = screen_info_l.texcoords;

    u32 size = sizeof(ScreenRectVertex) * 4;
    auto [ptr, offset, invalidate] = vertex_buffer.Map(size, 16);

    const std::array vertices = {ScreenRectVertex{x, y, texcoords.bottom, texcoords.left},
                                 ScreenRectVertex{x + w, y, texcoords.bottom, texcoords.right},
                                 ScreenRectVertex{x, y + h, texcoords.top, texcoords.left},
                                 ScreenRectVertex{x + w, y + h, texcoords.top, texcoords.right}};

    std::memcpy(ptr, vertices.data(), size);
    vertex_buffer.Commit(size);

    const u16 scale_factor = VideoCore::GetResolutionScaleFactor();
    const float width = static_cast<float>(screen_info_l.texture.width);
    const float height = static_cast<float>(screen_info_l.texture.height);

    draw_info.i_resolution =
        Common::Vec4f{width * scale_factor, height * scale_factor, 1.0f / (width * scale_factor),
                      1.0f / (height * scale_factor)};

    draw_info.o_resolution = Common::Vec4f{h, w, 1.0f / h, 1.0f / w};
    draw_info.screen_id_l = screen_id_l;
    draw_info.screen_id_r = screen_id_r;

    scheduler.Record([this, offset = offset, info = draw_info](vk::CommandBuffer cmdbuf) {
        cmdbuf.pushConstants(present_pipeline_layout,
                             vk::ShaderStageFlagBits::eFragment | vk::ShaderStageFlagBits::eVertex,
                             0, sizeof(info), &info);

        cmdbuf.bindVertexBuffers(0, vertex_buffer.Handle(), {0});
        cmdbuf.draw(4, 1, offset / sizeof(ScreenRectVertex), 0);
    });
}

void RendererVulkan::DrawSingleScreenStereo(u32 screen_id_l, u32 screen_id_r, float x, float y,
                                            float w, float h) {
    const ScreenInfo& screen_info_l = screen_infos[screen_id_l];
    const auto& texcoords = screen_info_l.texcoords;

    u32 size = sizeof(ScreenRectVertex) * 4;
    auto [ptr, offset, invalidate] = vertex_buffer.Map(size, 16);

    const std::array<ScreenRectVertex, 4> vertices = {{
        ScreenRectVertex(x, y, texcoords.bottom, texcoords.right),
        ScreenRectVertex(x + w, y, texcoords.top, texcoords.right),
        ScreenRectVertex(x, y + h, texcoords.bottom, texcoords.left),
        ScreenRectVertex(x + w, y + h, texcoords.top, texcoords.left),
    }};

    std::memcpy(ptr, vertices.data(), size);
    vertex_buffer.Commit(size);

    const u16 scale_factor = VideoCore::GetResolutionScaleFactor();
    const float width = static_cast<float>(screen_info_l.texture.width);
    const float height = static_cast<float>(screen_info_l.texture.height);

    draw_info.i_resolution =
        Common::Vec4f{width * scale_factor, height * scale_factor, 1.0f / (width * scale_factor),
                      1.0f / (height * scale_factor)};

    draw_info.o_resolution = Common::Vec4f{w, h, 1.0f / w, 1.0f / h};
    draw_info.screen_id_l = screen_id_l;
    draw_info.screen_id_r = screen_id_r;

    scheduler.Record([this, offset = offset, info = draw_info](vk::CommandBuffer cmdbuf) {
        cmdbuf.pushConstants(present_pipeline_layout,
                             vk::ShaderStageFlagBits::eFragment | vk::ShaderStageFlagBits::eVertex,
                             0, sizeof(info), &info);

        cmdbuf.bindVertexBuffers(0, vertex_buffer.Handle(), {0});
        cmdbuf.draw(4, 1, offset / sizeof(ScreenRectVertex), 0);
    });
}

void RendererVulkan::DrawScreens(Frame* frame, const Layout::FramebufferLayout& layout,
                                 bool flipped) {
    if (settings.bg_color_update_requested.exchange(false)) {
        clear_color.float32[0] = Settings::values.bg_red.GetValue();
        clear_color.float32[1] = Settings::values.bg_green.GetValue();
        clear_color.float32[2] = Settings::values.bg_blue.GetValue();
    }
    if (settings.sampler_update_requested.exchange(false)) {
        ReloadSampler();
    }
    if (settings.shader_update_requested.exchange(false)) {
        ReloadPipeline();
    }

    const auto& top_screen = layout.top_screen;
    const auto& bottom_screen = layout.bottom_screen;

    // Set projection matrix
    // draw_info.modelview =
    //    MakeOrthographicMatrix(static_cast<float>(layout.width),
    //    static_cast<float>(layout.height), flipped);
    draw_info.modelview = glm::transpose(glm::ortho(
        0.f, static_cast<float>(layout.width), static_cast<float>(layout.height), 0.0f, 0.f, 1.f));

    const Settings::StereoRenderOption render_3d = Settings::values.render_3d.GetValue();
    const bool stereo_single_screen = render_3d == Settings::StereoRenderOption::Anaglyph ||
                                      render_3d == Settings::StereoRenderOption::Interlaced ||
                                      render_3d == Settings::StereoRenderOption::ReverseInterlaced;

    // Bind necessary state before drawing the screens
    BeginRendering(frame);

    draw_info.layer = 0;
    if (layout.top_screen_enabled) {
        if (layout.is_rotated) {
            if (render_3d == Settings::StereoRenderOption::Off) {
                int eye = static_cast<int>(Settings::values.mono_render_option.GetValue());
                DrawSingleScreenRotated(eye, top_screen.left, top_screen.top, top_screen.GetWidth(),
                                        top_screen.GetHeight());
            } else if (render_3d == Settings::StereoRenderOption::SideBySide) {
                DrawSingleScreenRotated(0, (float)top_screen.left / 2, (float)top_screen.top,
                                        (float)top_screen.GetWidth() / 2,
                                        (float)top_screen.GetHeight());
                draw_info.layer = 1;
                DrawSingleScreenRotated(1, ((float)top_screen.left / 2) + ((float)layout.width / 2),
                                        (float)top_screen.top, (float)top_screen.GetWidth() / 2,
                                        (float)top_screen.GetHeight());
            } else if (render_3d == Settings::StereoRenderOption::CardboardVR) {
                DrawSingleScreenRotated(0, layout.top_screen.left, layout.top_screen.top,
                                        layout.top_screen.GetWidth(),
                                        layout.top_screen.GetHeight());
                draw_info.layer = 1;
                DrawSingleScreenRotated(
                    1, layout.cardboard.top_screen_right_eye + ((float)layout.width / 2),
                    layout.top_screen.top, layout.top_screen.GetWidth(),
                    layout.top_screen.GetHeight());
            } else if (stereo_single_screen) {
                DrawSingleScreenStereoRotated(0, 1, (float)top_screen.left, (float)top_screen.top,
                                              (float)top_screen.GetWidth(),
                                              (float)top_screen.GetHeight());
            }
        } else {
            if (render_3d == Settings::StereoRenderOption::Off) {
                int eye = static_cast<int>(Settings::values.mono_render_option.GetValue());
                DrawSingleScreen(eye, (float)top_screen.left, (float)top_screen.top,
                                 (float)top_screen.GetWidth(), (float)top_screen.GetHeight());
            } else if (render_3d == Settings::StereoRenderOption::SideBySide) {
                DrawSingleScreen(0, (float)top_screen.left / 2, (float)top_screen.top,
                                 (float)top_screen.GetWidth() / 2, (float)top_screen.GetHeight());
                draw_info.layer = 1;
                DrawSingleScreen(1, ((float)top_screen.left / 2) + ((float)layout.width / 2),
                                 (float)top_screen.top, (float)top_screen.GetWidth() / 2,
                                 (float)top_screen.GetHeight());
            } else if (render_3d == Settings::StereoRenderOption::CardboardVR) {
                DrawSingleScreen(0, layout.top_screen.left, layout.top_screen.top,
                                 layout.top_screen.GetWidth(), layout.top_screen.GetHeight());
                draw_info.layer = 1;
                DrawSingleScreen(1,
                                 layout.cardboard.top_screen_right_eye + ((float)layout.width / 2),
                                 layout.top_screen.top, layout.top_screen.GetWidth(),
                                 layout.top_screen.GetHeight());
            } else if (stereo_single_screen) {
                DrawSingleScreenStereo(0, 1, (float)top_screen.left, (float)top_screen.top,
                                       (float)top_screen.GetWidth(), (float)top_screen.GetHeight());
            }
        }
    }

    draw_info.layer = 0;
    if (layout.bottom_screen_enabled) {
        if (layout.is_rotated) {
            if (render_3d == Settings::StereoRenderOption::Off) {
                DrawSingleScreenRotated(2, (float)bottom_screen.left, (float)bottom_screen.top,
                                        (float)bottom_screen.GetWidth(),
                                        (float)bottom_screen.GetHeight());
            } else if (render_3d == Settings::StereoRenderOption::SideBySide) {
                DrawSingleScreenRotated(2, (float)bottom_screen.left / 2, (float)bottom_screen.top,
                                        (float)bottom_screen.GetWidth() / 2,
                                        (float)bottom_screen.GetHeight());
                draw_info.layer = 1;
                DrawSingleScreenRotated(
                    2, ((float)bottom_screen.left / 2) + ((float)layout.width / 2),
                    (float)bottom_screen.top, (float)bottom_screen.GetWidth() / 2,
                    (float)bottom_screen.GetHeight());
            } else if (render_3d == Settings::StereoRenderOption::CardboardVR) {
                DrawSingleScreenRotated(2, layout.bottom_screen.left, layout.bottom_screen.top,
                                        layout.bottom_screen.GetWidth(),
                                        layout.bottom_screen.GetHeight());
                draw_info.layer = 1;
                DrawSingleScreenRotated(
                    2, layout.cardboard.bottom_screen_right_eye + ((float)layout.width / 2),
                    layout.bottom_screen.top, layout.bottom_screen.GetWidth(),
                    layout.bottom_screen.GetHeight());
            } else if (stereo_single_screen) {
                DrawSingleScreenStereoRotated(
                    2, 2, (float)bottom_screen.left, (float)bottom_screen.top,
                    (float)bottom_screen.GetWidth(), (float)bottom_screen.GetHeight());
            }
        } else {
            if (render_3d == Settings::StereoRenderOption::Off) {
                DrawSingleScreen(2, (float)bottom_screen.left, (float)bottom_screen.top,
                                 (float)bottom_screen.GetWidth(), (float)bottom_screen.GetHeight());
            } else if (render_3d == Settings::StereoRenderOption::SideBySide) {
                DrawSingleScreen(2, (float)bottom_screen.left / 2, (float)bottom_screen.top,
                                 (float)bottom_screen.GetWidth() / 2,
                                 (float)bottom_screen.GetHeight());
                draw_info.layer = 1;
                DrawSingleScreen(2, ((float)bottom_screen.left / 2) + ((float)layout.width / 2),
                                 (float)bottom_screen.top, (float)bottom_screen.GetWidth() / 2,
                                 (float)bottom_screen.GetHeight());
            } else if (render_3d == Settings::StereoRenderOption::CardboardVR) {
                DrawSingleScreen(2, layout.bottom_screen.left, layout.bottom_screen.top,
                                 layout.bottom_screen.GetWidth(), layout.bottom_screen.GetHeight());
                draw_info.layer = 1;
                DrawSingleScreen(
                    2, layout.cardboard.bottom_screen_right_eye + ((float)layout.width / 2),
                    layout.bottom_screen.top, layout.bottom_screen.GetWidth(),
                    layout.bottom_screen.GetHeight());
            } else if (stereo_single_screen) {
                DrawSingleScreenStereo(2, 2, (float)bottom_screen.left, (float)bottom_screen.top,
                                       (float)bottom_screen.GetWidth(),
                                       (float)bottom_screen.GetHeight());
            }
        }
    }

    scheduler.Record([image = frame->image](vk::CommandBuffer cmdbuf) {
        const vk::ImageMemoryBarrier render_barrier = {
            .srcAccessMask = vk::AccessFlagBits::eColorAttachmentWrite,
            .dstAccessMask = vk::AccessFlagBits::eTransferRead,
            .oldLayout = vk::ImageLayout::eTransferSrcOptimal,
            .newLayout = vk::ImageLayout::eTransferSrcOptimal,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .image = image,
            .subresourceRange{
                .aspectMask = vk::ImageAspectFlagBits::eColor,
                .baseMipLevel = 0,
                .levelCount = 1,
                .baseArrayLayer = 0,
                .layerCount = VK_REMAINING_ARRAY_LAYERS,
            },
        };

        cmdbuf.endRenderPass();
        cmdbuf.pipelineBarrier(vk::PipelineStageFlagBits::eColorAttachmentOutput,
                               vk::PipelineStageFlagBits::eTransfer,
                               vk::DependencyFlagBits::eByRegion, {}, {}, render_barrier);
    });
}

void RendererVulkan::SwapBuffers() {
    bool has_been_recreated = false;
    const auto recreate_swapchain = [&](u32 width, u32 height) {
        if (!has_been_recreated) {
            has_been_recreated = true;
            scheduler.Finish();
        }
        swapchain.Create(width, height);
    };
    const auto& layout = render_window.GetFramebufferLayout();
    if (swapchain.NeedsRecreation() || swapchain.GetWidth() != layout.width ||
        swapchain.GetHeight() != layout.height) {
        recreate_swapchain(layout.width, layout.height);
    }

    PrepareRendertarget();
    RenderScreenshot();

    RenderToMailbox(layout, mailbox, false);

    m_current_frame++;

    system.perf_stats->EndSystemFrame();
    render_window.PollEvents();

    system.frame_limiter.DoFrameLimiting(system.CoreTiming().GetGlobalTimeUs());
    system.perf_stats->BeginSystemFrame();

    if (Pica::g_debug_context && Pica::g_debug_context->recorder) {
        Pica::g_debug_context->recorder->FrameFinished();
    }
}

void RendererVulkan::RenderScreenshot() {
    if (!settings.screenshot_requested.exchange(false)) {
        return;
    }

    const Layout::FramebufferLayout layout{settings.screenshot_framebuffer_layout};
    const vk::Extent2D extent = swapchain.GetExtent();
    const u32 width = std::min(layout.width, extent.width);
    const u32 height = std::min(layout.height, extent.height);

    const vk::ImageCreateInfo staging_image_info = {
        .imageType = vk::ImageType::e2D,
        .format = vk::Format::eB8G8R8A8Unorm,
        .extent{
            .width = width,
            .height = height,
            .depth = 1,
        },
        .mipLevels = 1,
        .arrayLayers = 1,
        .samples = vk::SampleCountFlagBits::e1,
        .tiling = vk::ImageTiling::eLinear,
        .usage = vk::ImageUsageFlagBits::eTransferDst,
        .initialLayout = vk::ImageLayout::eUndefined,
    };

    const VmaAllocationCreateInfo alloc_create_info = {
        .flags = VMA_ALLOCATION_CREATE_WITHIN_BUDGET_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT |
                 VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT,
        .usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST,
        .requiredFlags = 0,
        .preferredFlags = 0,
        .pool = VK_NULL_HANDLE,
        .pUserData = nullptr,
    };

    VkImage unsafe_image{};
    VmaAllocation allocation{};
    VmaAllocationInfo alloc_info;
    VkImageCreateInfo unsafe_image_info = static_cast<VkImageCreateInfo>(staging_image_info);

    VkResult result = vmaCreateImage(instance.GetAllocator(), &unsafe_image_info,
                                     &alloc_create_info, &unsafe_image, &allocation, &alloc_info);
    if (result != VK_SUCCESS) [[unlikely]] {
        LOG_CRITICAL(Render_Vulkan, "Failed allocating texture with error {}", result);
        UNREACHABLE();
    }
    vk::Image staging_image{unsafe_image};

    Frame frame{};
    mailbox->ReloadFrame(&frame, width, height);

    DrawScreens(&frame, layout, false);

    scheduler.Record(
        [width, height, source_image = frame.image, staging_image](vk::CommandBuffer cmdbuf) {
            const std::array read_barriers = {
                vk::ImageMemoryBarrier{
                    .srcAccessMask = vk::AccessFlagBits::eMemoryWrite,
                    .dstAccessMask = vk::AccessFlagBits::eTransferRead,
                    .oldLayout = vk::ImageLayout::eTransferSrcOptimal,
                    .newLayout = vk::ImageLayout::eTransferSrcOptimal,
                    .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                    .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                    .image = source_image,
                    .subresourceRange{
                        .aspectMask = vk::ImageAspectFlagBits::eColor,
                        .baseMipLevel = 0,
                        .levelCount = VK_REMAINING_MIP_LEVELS,
                        .baseArrayLayer = 0,
                        .layerCount = VK_REMAINING_ARRAY_LAYERS,
                    },
                },
                vk::ImageMemoryBarrier{
                    .srcAccessMask = vk::AccessFlagBits::eNone,
                    .dstAccessMask = vk::AccessFlagBits::eTransferWrite,
                    .oldLayout = vk::ImageLayout::eUndefined,
                    .newLayout = vk::ImageLayout::eTransferDstOptimal,
                    .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                    .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                    .image = staging_image,
                    .subresourceRange{
                        .aspectMask = vk::ImageAspectFlagBits::eColor,
                        .baseMipLevel = 0,
                        .levelCount = VK_REMAINING_MIP_LEVELS,
                        .baseArrayLayer = 0,
                        .layerCount = VK_REMAINING_ARRAY_LAYERS,
                    },
                },
            };
            const std::array write_barriers = {
                vk::ImageMemoryBarrier{
                    .srcAccessMask = vk::AccessFlagBits::eTransferRead,
                    .dstAccessMask = vk::AccessFlagBits::eMemoryWrite,
                    .oldLayout = vk::ImageLayout::eTransferSrcOptimal,
                    .newLayout = vk::ImageLayout::eTransferSrcOptimal,
                    .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                    .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                    .image = source_image,
                    .subresourceRange{
                        .aspectMask = vk::ImageAspectFlagBits::eColor,
                        .baseMipLevel = 0,
                        .levelCount = VK_REMAINING_MIP_LEVELS,
                        .baseArrayLayer = 0,
                        .layerCount = VK_REMAINING_ARRAY_LAYERS,
                    },
                },
                vk::ImageMemoryBarrier{
                    .srcAccessMask = vk::AccessFlagBits::eTransferWrite,
                    .dstAccessMask = vk::AccessFlagBits::eMemoryRead,
                    .oldLayout = vk::ImageLayout::eTransferDstOptimal,
                    .newLayout = vk::ImageLayout::eGeneral,
                    .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                    .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                    .image = staging_image,
                    .subresourceRange{
                        .aspectMask = vk::ImageAspectFlagBits::eColor,
                        .baseMipLevel = 0,
                        .levelCount = VK_REMAINING_MIP_LEVELS,
                        .baseArrayLayer = 0,
                        .layerCount = VK_REMAINING_ARRAY_LAYERS,
                    },
                },
            };
            static constexpr vk::MemoryBarrier memory_write_barrier = {
                .srcAccessMask = vk::AccessFlagBits::eMemoryWrite,
                .dstAccessMask = vk::AccessFlagBits::eMemoryRead | vk::AccessFlagBits::eMemoryWrite,
            };

            const std::array offsets = {
                vk::Offset3D{0, 0, 0},
                vk::Offset3D{static_cast<s32>(width), static_cast<s32>(height), 1},
            };

            const vk::ImageBlit blit_area = {
                .srcSubresource{
                    .aspectMask = vk::ImageAspectFlagBits::eColor,
                    .mipLevel = 0,
                    .baseArrayLayer = 0,
                    .layerCount = 1,
                },
                .srcOffsets = offsets,
                .dstSubresource{
                    .aspectMask = vk::ImageAspectFlagBits::eColor,
                    .mipLevel = 0,
                    .baseArrayLayer = 0,
                    .layerCount = 1,
                },
                .dstOffsets = offsets,
            };

            cmdbuf.pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands,
                                   vk::PipelineStageFlagBits::eTransfer,
                                   vk::DependencyFlagBits::eByRegion, {}, {}, read_barriers);
            cmdbuf.blitImage(source_image, vk::ImageLayout::eTransferSrcOptimal, staging_image,
                             vk::ImageLayout::eTransferDstOptimal, blit_area, vk::Filter::eNearest);
            cmdbuf.pipelineBarrier(
                vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eAllCommands,
                vk::DependencyFlagBits::eByRegion, memory_write_barrier, {}, write_barriers);
        });

    // Ensure the copy is fully completed before saving the screenshot
    scheduler.Finish();

    const vk::Device device = instance.GetDevice();

    // Get layout of the image (including row pitch)
    const vk::ImageSubresource subresource = {
        .aspectMask = vk::ImageAspectFlagBits::eColor,
        .mipLevel = 0,
        .arrayLayer = 0,
    };

    const vk::SubresourceLayout subresource_layout =
        device.getImageSubresourceLayout(staging_image, subresource);

    // Copy backing image data to the QImage screenshot buffer
    const u8* data = reinterpret_cast<const u8*>(alloc_info.pMappedData);
    std::memcpy(settings.screenshot_bits, data + subresource_layout.offset,
                subresource_layout.size);

    // Destroy allocated resources
    vmaDestroyImage(instance.GetAllocator(), frame.image, frame.allocation);
    device.destroyFramebuffer(frame.framebuffer);
    device.destroyImageView(frame.image_view);

    settings.screenshot_complete_callback();
}

void RendererVulkan::NotifySurfaceChanged() {
    scheduler.Finish();
    vk::SurfaceKHR new_surface = CreateSurface(instance.GetInstance(), render_window);
    mailbox->UpdateSurface(render_window.GetFramebufferLayout().width,
                           render_window.GetFramebufferLayout().height, new_surface);
}

void RendererVulkan::Report() const {
    const std::string vendor_name{instance.GetVendorName()};
    const std::string model_name{instance.GetModelName()};
    const std::string driver_version = GetDriverVersion(instance);
    const std::string driver_name = fmt::format("{} {}", vendor_name, driver_version);

    const std::string api_version = GetReadableVersion(instance.ApiVersion());

    const std::string extensions = BuildCommaSeparatedExtensions(instance.GetAvailableExtensions());

    LOG_INFO(Render_Vulkan, "Driver: {}", driver_name);
    LOG_INFO(Render_Vulkan, "Device: {}", model_name);
    LOG_INFO(Render_Vulkan, "Vulkan: {}", api_version);

    static constexpr auto field = Common::Telemetry::FieldType::UserSystem;
    telemetry_session.AddField(field, "GPU_Vendor", vendor_name);
    telemetry_session.AddField(field, "GPU_Model", model_name);
    telemetry_session.AddField(field, "GPU_Vulkan_Driver", driver_name);
    telemetry_session.AddField(field, "GPU_Vulkan_Version", api_version);
    telemetry_session.AddField(field, "GPU_Vulkan_Extensions", extensions);
}

} // namespace Vulkan
