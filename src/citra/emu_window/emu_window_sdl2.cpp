// Copyright 2016 Citra Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#include <algorithm>
#include <cstdlib>
#include <string>
#define SDL_MAIN_HANDLED
#include <SDL.h>
#include <fmt/format.h>
#include <glad/glad.h>
#include "citra/emu_window/emu_window_sdl2.h"
#include "common/logging/log.h"
#include "common/scm_rev.h"
#include "common/settings.h"
#include "core/3ds.h"
#include "core/core.h"
#include "input_common/keyboard.h"
#include "input_common/main.h"
#include "input_common/motion_emu.h"
#include "input_common/sdl/sdl.h"
#include "network/network.h"
#include "video_core/renderer_base.h"
#include "video_core/video_core.h"

SharedContext_SDL2::SharedContext_SDL2() {
    window = SDL_CreateWindow(NULL, SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED, 0, 0,
                              SDL_WINDOW_HIDDEN | SDL_WINDOW_OPENGL);
    context = SDL_GL_CreateContext(window);
}

SharedContext_SDL2::~SharedContext_SDL2() {
    SDL_GL_DeleteContext(context);
    SDL_DestroyWindow(window);
}

void SharedContext_SDL2::MakeCurrent() {
    SDL_GL_MakeCurrent(window, context);
}

void SharedContext_SDL2::DoneCurrent() {
    SDL_GL_MakeCurrent(window, nullptr);
}

void EmuWindow_SDL2::OnMouseMotion(s32 x, s32 y) {
    TouchMoved((unsigned)std::max(x, 0), (unsigned)std::max(y, 0));
    InputCommon::GetMotionEmu()->Tilt(x, y);
}

void EmuWindow_SDL2::OnMouseButton(u32 button, u8 state, s32 x, s32 y) {
    if (button == SDL_BUTTON_LEFT) {
        if (state == SDL_PRESSED) {
            TouchPressed((unsigned)std::max(x, 0), (unsigned)std::max(y, 0));
        } else {
            TouchReleased();
        }
    } else if (button == SDL_BUTTON_RIGHT) {
        if (state == SDL_PRESSED) {
            InputCommon::GetMotionEmu()->BeginTilt(x, y);
        } else {
            InputCommon::GetMotionEmu()->EndTilt();
        }
    }
}

std::pair<unsigned, unsigned> EmuWindow_SDL2::TouchToPixelPos(float touch_x, float touch_y) const {
    int w, h;
    SDL_GL_GetDrawableSize(render_window, &w, &h);

    touch_x *= w;
    touch_y *= h;

    return {static_cast<unsigned>(std::max(std::round(touch_x), 0.0f)),
            static_cast<unsigned>(std::max(std::round(touch_y), 0.0f))};
}

void EmuWindow_SDL2::OnFingerDown(float x, float y) {
    // TODO(NeatNit): keep track of multitouch using the fingerID and a dictionary of some kind
    // This isn't critical because the best we can do when we have that is to average them, like the
    // 3DS does

    const auto [px, py] = TouchToPixelPos(x, y);
    TouchPressed(px, py);
}

void EmuWindow_SDL2::OnFingerMotion(float x, float y) {
    const auto [px, py] = TouchToPixelPos(x, y);
    TouchMoved(px, py);
}

void EmuWindow_SDL2::OnFingerUp() {
    TouchReleased();
}

void EmuWindow_SDL2::OnKeyEvent(int key, u8 state) {
    if (state == SDL_PRESSED) {
        InputCommon::GetKeyboard()->PressKey(key);
    } else if (state == SDL_RELEASED) {
        InputCommon::GetKeyboard()->ReleaseKey(key);
    }
}

bool EmuWindow_SDL2::IsOpen() const {
    return is_open;
}

void EmuWindow_SDL2::RequestClose() {
    is_open = false;
}

void EmuWindow_SDL2::OnResize() {
    int width, height;
    SDL_GL_GetDrawableSize(render_window, &width, &height);
    UpdateCurrentFramebufferLayout(width, height);
}

void EmuWindow_SDL2::Fullscreen() {
    if (SDL_SetWindowFullscreen(render_window, SDL_WINDOW_FULLSCREEN) == 0) {
        return;
    }

    LOG_ERROR(Frontend, "Fullscreening failed: {}", SDL_GetError());

    // Try a different fullscreening method
    LOG_INFO(Frontend, "Attempting to use borderless fullscreen...");
    if (SDL_SetWindowFullscreen(render_window, SDL_WINDOW_FULLSCREEN_DESKTOP) == 0) {
        return;
    }

    LOG_ERROR(Frontend, "Borderless fullscreening failed: {}", SDL_GetError());

    // Fallback algorithm: Maximise window.
    // Works on all systems (unless something is seriously wrong), so no fallback for this one.
    LOG_INFO(Frontend, "Falling back on a maximised window...");
    SDL_MaximizeWindow(render_window);
}

EmuWindow_SDL2::EmuWindow_SDL2(bool fullscreen, bool is_secondary) : EmuWindow(is_secondary) {
    // Initialize the window
    const bool is_opengles =
        Settings::values.graphics_api.GetValue() == Settings::GraphicsAPI::OpenGLES;
    if (is_opengles) {
        SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 3);
        SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 2);
        SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_ES);
    } else {
        SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 4);
        SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 3);
        SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_CORE);
    }

    SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, 1);
    SDL_GL_SetAttribute(SDL_GL_RED_SIZE, 8);
    SDL_GL_SetAttribute(SDL_GL_GREEN_SIZE, 8);
    SDL_GL_SetAttribute(SDL_GL_BLUE_SIZE, 8);
    SDL_GL_SetAttribute(SDL_GL_ALPHA_SIZE, 0);
    // Enable context sharing for the shared context
    SDL_GL_SetAttribute(SDL_GL_SHARE_WITH_CURRENT_CONTEXT, 1);
    // Enable vsync
    SDL_GL_SetSwapInterval(1);

    std::string window_title = fmt::format("Citra {} | {}-{}", Common::g_build_fullname,
                                           Common::g_scm_branch, Common::g_scm_desc);
    render_window =
        SDL_CreateWindow(window_title.c_str(),
                         SDL_WINDOWPOS_UNDEFINED, // x position
                         SDL_WINDOWPOS_UNDEFINED, // y position
                         Core::kScreenTopWidth, Core::kScreenTopHeight + Core::kScreenBottomHeight,
                         SDL_WINDOW_OPENGL | SDL_WINDOW_RESIZABLE | SDL_WINDOW_ALLOW_HIGHDPI);

    if (render_window == nullptr) {
        LOG_CRITICAL(Frontend, "Failed to create SDL2 window: {}", SDL_GetError());
        exit(1);
    }

    strict_context_required = std::strcmp(SDL_GetCurrentVideoDriver(), "wayland") == 0;

    dummy_window = SDL_CreateWindow(NULL, SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED, 0, 0,
                                    SDL_WINDOW_HIDDEN | SDL_WINDOW_OPENGL);

    if (fullscreen) {
        Fullscreen();
    }

    window_context = SDL_GL_CreateContext(render_window);
    core_context = CreateSharedContext();
    last_saved_context = nullptr;

    if (window_context == nullptr) {
        LOG_CRITICAL(Frontend, "Failed to create SDL2 GL context: {}", SDL_GetError());
        exit(1);
    }
    if (core_context == nullptr) {
        LOG_CRITICAL(Frontend, "Failed to create shared SDL2 GL context: {}", SDL_GetError());
        exit(1);
    }

    render_window_id = SDL_GetWindowID(render_window);
    auto gl_load_func = is_opengles ? gladLoadGLES2Loader : gladLoadGLLoader;

    if (!gl_load_func(static_cast<GLADloadproc>(SDL_GL_GetProcAddress))) {
        LOG_CRITICAL(Frontend, "Failed to initialize GL functions: {}", SDL_GetError());
        exit(1);
    }

    OnResize();
    OnMinimalClientAreaChangeRequest(GetActiveConfig().min_client_area_size);
    SDL_PumpEvents();
}

EmuWindow_SDL2::~EmuWindow_SDL2() {
    core_context.reset();
    SDL_GL_DeleteContext(window_context);
    SDL_Quit();
}

void EmuWindow_SDL2::InitializeSDL2() {
    if (SDL_Init(SDL_INIT_VIDEO | SDL_INIT_GAMECONTROLLER) < 0) {
        LOG_CRITICAL(Frontend, "Failed to initialize SDL2: {}! Exiting...", SDL_GetError());
        exit(1);
    }

    InputCommon::Init();
    Network::Init();

    SDL_SetMainReady();
}

std::unique_ptr<Frontend::GraphicsContext> EmuWindow_SDL2::CreateSharedContext() const {
    return std::make_unique<SharedContext_SDL2>();
}

void EmuWindow_SDL2::SaveContext() {
    last_saved_context = SDL_GL_GetCurrentContext();
}

void EmuWindow_SDL2::RestoreContext() {
    SDL_GL_MakeCurrent(render_window, last_saved_context);
}

void EmuWindow_SDL2::Present() {
    SDL_GL_MakeCurrent(render_window, window_context);
    SDL_GL_SetSwapInterval(1);
    while (IsOpen()) {
        VideoCore::g_renderer->TryPresent(100, is_secondary);
        SDL_GL_SwapWindow(render_window);
    }
    SDL_GL_MakeCurrent(render_window, nullptr);
}

void EmuWindow_SDL2::PollEvents() {
    SDL_Event event;
    std::vector<SDL_Event> other_window_events;

    // SDL_PollEvent returns 0 when there are no more events in the event queue
    while (SDL_PollEvent(&event)) {
        if (event.window.windowID != render_window_id) {
            other_window_events.push_back(event);
            continue;
        }
        switch (event.type) {
        case SDL_WINDOWEVENT:
            switch (event.window.event) {
            case SDL_WINDOWEVENT_SIZE_CHANGED:
            case SDL_WINDOWEVENT_RESIZED:
            case SDL_WINDOWEVENT_MAXIMIZED:
            case SDL_WINDOWEVENT_RESTORED:
            case SDL_WINDOWEVENT_MINIMIZED:
                OnResize();
                break;
            case SDL_WINDOWEVENT_CLOSE:
                RequestClose();
                break;
            }
            break;
        case SDL_KEYDOWN:
        case SDL_KEYUP:
            OnKeyEvent(static_cast<int>(event.key.keysym.scancode), event.key.state);
            break;
        case SDL_MOUSEMOTION:
            // ignore if it came from touch
            if (event.button.which != SDL_TOUCH_MOUSEID)
                OnMouseMotion(event.motion.x, event.motion.y);
            break;
        case SDL_MOUSEBUTTONDOWN:
        case SDL_MOUSEBUTTONUP:
            // ignore if it came from touch
            if (event.button.which != SDL_TOUCH_MOUSEID) {
                OnMouseButton(event.button.button, event.button.state, event.button.x,
                              event.button.y);
            }
            break;
        case SDL_FINGERDOWN:
            OnFingerDown(event.tfinger.x, event.tfinger.y);
            break;
        case SDL_FINGERMOTION:
            OnFingerMotion(event.tfinger.x, event.tfinger.y);
            break;
        case SDL_FINGERUP:
            OnFingerUp();
            break;
        case SDL_QUIT:
            RequestClose();
            break;
        default:
            break;
        }
    }
    for (auto& e : other_window_events) {
        // This is a somewhat hacky workaround to re-emit window events meant for another window
        // since SDL_PollEvent() is global but we poll events per window.
        SDL_PushEvent(&e);
    }
    if (!is_secondary) {
        UpdateFramerateCounter();
    }
}

void EmuWindow_SDL2::MakeCurrent() {
    core_context->MakeCurrent();
}

void EmuWindow_SDL2::DoneCurrent() {
    core_context->DoneCurrent();
}

void EmuWindow_SDL2::OnMinimalClientAreaChangeRequest(std::pair<u32, u32> minimal_size) {
    SDL_SetWindowMinimumSize(render_window, minimal_size.first, minimal_size.second);
}

void EmuWindow_SDL2::UpdateFramerateCounter() {
    const u32 current_time = SDL_GetTicks();
    if (current_time > last_time + 2000) {
        const auto results = Core::System::GetInstance().GetAndResetPerfStats();
        const auto title =
            fmt::format("Citra {} | {}-{} | FPS: {:.0f} ({:.0f}%)", Common::g_build_fullname,
                        Common::g_scm_branch, Common::g_scm_desc, results.game_fps,
                        results.emulation_speed * 100.0f);
        SDL_SetWindowTitle(render_window, title.c_str());
        last_time = current_time;
    }
}
