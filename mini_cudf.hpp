#pragma once
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <utility>

// 1. Memory Manager (No change)
struct DeviceBuffer{
    void* ptr=nullptr;
    size_t bytes=0;

    DeviceBuffer()=default;
    DeviceBuffer(size_t n):bytes(n){
        if(n>0) cudaMalloc(&ptr,n);
    }
    ~DeviceBuffer(){
        if(ptr!=nullptr) cudaFree(ptr);
    }

    DeviceBuffer(const DeviceBuffer&)=delete;

    DeviceBuffer(DeviceBuffer&& other) noexcept {
        ptr=other.ptr;
        bytes=other.bytes;
        other.ptr=nullptr;
        other.bytes=0;
    }

    template<typename T>
    T* data(){ return reinterpret_cast<T*>(ptr); }

    template<typename T>
    const T* data() const { return reinterpret_cast<const T*>(ptr); }
};

// 2. Mehmaan Chashma (No change)
struct StringView{
    const int* d_offsets;
    const char* d_chars;
    int num_strings;
    int total_bytes;
};

// 3. NAYA: Poore Column ka Maalik (Offsets + Chars dono ko pack karta hai!)
struct StringsColumn{
    DeviceBuffer offsets;
    DeviceBuffer chars;
    int num_strings=0;
    int total_bytes=0;

    // Constructor: Move Semantics (&&) se dono buffers ka maalik banta hai
    StringsColumn(DeviceBuffer&& off,DeviceBuffer&& ch,int n_str,int n_bytes)
        :offsets(std::move(off)),chars(std::move(ch)),num_strings(n_str),total_bytes(n_bytes){}

    // Is dabba se Mehmaan View nikalne ka shortcut:
    StringView view() const {
        return StringView{offsets.data<int>(),chars.data<char>(),num_strings,total_bytes};
    }
};

// 4. NAYA MENU CARD: Ab yeh function sidha StringsColumn return karega!
StringsColumn reverse_strings(StringView in);
