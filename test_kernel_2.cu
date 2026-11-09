#include <cuda_runtime.h>
#include <cstdio>
#include <utility>

// 1. AII Memory Buffer 
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
};

// 2. 24-byte ka StringView
struct StringView{
    const int* d_offsets; // Address 0x100
    const char* d_chars;   // Address 0x200
    int num_strings;       // 2
    int total_bytes;       // 6
};

// 3. Kernel
__global__ void reverse_strings_kernel(StringView in,char* d_out){
    int idx=blockIdx.x*blockDim.x+threadIdx.x;
    if(idx>=in.num_strings) return;

    // in lifafe ke andar se offsets aur chars nikaale
    int start=in.d_offsets[idx];
    int end=in.d_offsets[idx+1];
    int len=end-start;

    for(int k=0;k<len;k++){
        d_out[start+k]=in.d_chars[end-1-k];
    }
}

int main(){
    const char h_in[7]="CATDOG";
    const int h_offsets[3]={0,3,6};
    int num_strings=2;
    int total_bytes=6;

    // A. GPU Buffers banaye (DeviceBuffer se)
    DeviceBuffer buf_in(total_bytes*sizeof(char));
    DeviceBuffer buf_offsets((num_strings+1)*sizeof(int));
    DeviceBuffer buf_out(total_bytes*sizeof(char));

    cudaMemcpy(buf_in.data<char>(),h_in,total_bytes*sizeof(char),cudaMemcpyHostToDevice);
    cudaMemcpy(buf_offsets.data<int>(),h_offsets,(num_strings+1)*sizeof(int),cudaMemcpyHostToDevice);

    // B. CPU Stack par StringView ka 24-byte packet pack kiya
    StringView in_view{
        buf_offsets.data<int>(),
        buf_in.data<char>(),
        num_strings,
        total_bytes
    };

    // C. Clean Kernel Launch! (Sirf packet aur output address bheja)
    reverse_strings_kernel<<<1,2>>>(in_view,buf_out.data<char>());
    cudaDeviceSynchronize();

    // D. Print Result
    char h_out[7]={0};
    cudaMemcpy(h_out,buf_out.data<char>(),total_bytes*sizeof(char),cudaMemcpyDeviceToHost);
    printf("   STAGE 3 RUN: DeviceBuffer + StringView ON T4\n");
    printf("Input Strings   : \"CAT\", \"DOG\"\n");
    printf("Output Raw Chars: \"%s\"\n\n",h_out);

    for(int i=0;i<num_strings;i++){
        int start=h_offsets[i];
        int end=h_offsets[i+1];
        printf("String %d Reversed: \"",i);
        for(int k=start;k<end;k++){
            printf("%c",h_out[k]);
        }
        printf("\"\n");
    }
    printf("Clean StringView passed by value to GPU Registers!\n\n");

    return 0;
}
