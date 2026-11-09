#include "mini_cudf.hpp"

__global__ void reverse_strings_kernel(StringView in,char* d_out){
    int idx=blockIdx.x*blockDim.x+threadIdx.x;
    if(idx>=in.num_strings) return;

    int start=in.d_offsets[idx];
    int end=in.d_offsets[idx+1];
    int len=end-start;

    for(int k=0;k<len;k++){
        d_out[start+k]=in.d_chars[end-1-k];
    }
}

// NAYA: Ab yeh khud memory banata hai aur StringsColumn return karta hai!
StringsColumn reverse_strings(StringView in){
    // 1. Output ke naye chars ke liye GPU VRAM li
    DeviceBuffer out_chars(in.total_bytes*sizeof(char));

    // 2. Offsets nahi badle, toh unhe copy kiya (Device-to-Device: 0.001 ms)
    DeviceBuffer out_offsets((in.num_strings+1)*sizeof(int));
    cudaMemcpy(out_offsets.data<int>(),in.d_offsets,(in.num_strings+1)*sizeof(int),cudaMemcpyDeviceToDevice);

    // 3. Kernel Fire!
    reverse_strings_kernel<<<1,in.num_strings>>>(in,out_chars.data<char>());
    cudaDeviceSynchronize();

    // 4. Poora Column pack karke Move Semantics (&&) se return kiya!
    return StringsColumn(std::move(out_offsets),std::move(out_chars),in.num_strings,in.total_bytes);
}
