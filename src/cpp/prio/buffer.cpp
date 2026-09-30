// SPDX-License-Identifier: GPL-3.0-or-later
#include "prio/buffer.hpp"

#include <stdexcept>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#else
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cerrno>
#include <cstdlib>
#include <cstring>
#endif

namespace prio {

#if defined(_WIN32)

namespace {
std::wstring widen(const std::string& s) {
    if (s.empty()) return {};
    const int n = MultiByteToWideChar(CP_UTF8, 0, s.data(), static_cast<int>(s.size()), nullptr, 0);
    std::wstring w(static_cast<std::size_t>(n), L'\0');
    MultiByteToWideChar(CP_UTF8, 0, s.data(), static_cast<int>(s.size()), w.data(), n);
    return w;
}
[[noreturn]] void fail(const char* what) {
    throw std::runtime_error(std::string("temporary storage: ") + what + " failed (error " +
                             std::to_string(GetLastError()) + ")");
}
}  // namespace

MappedFile::MappedFile(const std::string& dir, std::size_t bytes) {
    wchar_t name[MAX_PATH + 1];
    if (!GetTempFileNameW(widen(dir).c_str(), L"lsp", 0, name)) fail("creating a file");
    // temporary (kept in the cache while memory allows) and deleted when closed
    file_ = CreateFileW(name, GENERIC_READ | GENERIC_WRITE, 0, nullptr, CREATE_ALWAYS,
                        FILE_ATTRIBUTE_TEMPORARY | FILE_FLAG_DELETE_ON_CLOSE, nullptr);
    if (file_ == INVALID_HANDLE_VALUE) {
        file_ = nullptr;
        DeleteFileW(name);
        fail("opening a file");
    }
    bytes_ = bytes;
    map();
}

MappedFile::~MappedFile() {
    unmap();
    if (file_) CloseHandle(file_);
}

void MappedFile::map() {
    const auto size = static_cast<unsigned long long>(bytes_);
    mapping_ = CreateFileMappingW(file_, nullptr, PAGE_READWRITE, static_cast<DWORD>(size >> 32),
                                  static_cast<DWORD>(size & 0xffffffffULL), nullptr);
    if (!mapping_) fail("mapping a file");
    data_ = MapViewOfFile(mapping_, FILE_MAP_ALL_ACCESS, 0, 0, 0);
    if (!data_) fail("mapping a file");
}

void MappedFile::unmap() {
    if (data_) UnmapViewOfFile(data_);
    if (mapping_) CloseHandle(mapping_);
    data_ = nullptr;
    mapping_ = nullptr;
}

void MappedFile::grow(std::size_t bytes) {
    if (bytes <= bytes_) return;
    unmap();
    bytes_ = bytes;
    map();
}

#else

namespace {
[[noreturn]] void fail(const char* what) {
    throw std::runtime_error(std::string("temporary storage: ") + what + " failed (" + std::strerror(errno) + ")");
}
}  // namespace

MappedFile::MappedFile(const std::string& dir, std::size_t bytes) {
    std::string path = (dir.empty() ? std::string(".") : dir) + "/lspXXXXXX";
    fd_ = mkstemp(path.data());
    if (fd_ < 0) fail("creating a file");
    unlink(path.c_str());  // deleted when closed
    bytes_ = bytes;
    map();
}

MappedFile::~MappedFile() {
    unmap();
    if (fd_ >= 0) close(fd_);
}

void MappedFile::map() {
    if (ftruncate(fd_, static_cast<off_t>(bytes_)) != 0) fail("sizing a file");
    void* p = mmap(nullptr, bytes_, PROT_READ | PROT_WRITE, MAP_SHARED, fd_, 0);
    if (p == MAP_FAILED) fail("mapping a file");
    data_ = p;
}

void MappedFile::unmap() {
    if (data_) munmap(data_, bytes_);
    data_ = nullptr;
}

void MappedFile::grow(std::size_t bytes) {
    if (bytes <= bytes_) return;
    unmap();
    bytes_ = bytes;
    map();
}

#endif

}  // namespace prio
