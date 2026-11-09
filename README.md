# cuda-string-engine

A small C++/CUDA project for handling variable-length strings on the GPU.

Strings are stored the Arrow way (one big character buffer plus an offsets array), GPU memory is wrapped in RAII types so it can't leak, and there is one working operation: reversing every string in a column.

I built this to understand how string columns are laid out on a GPU and how to keep the C++ side of CUDA code safe. I ran it on a Tesla T4 through Modal.

## How strings are stored

Giving every string its own GPU allocation is slow and fragments memory. Instead, all strings live in two flat arrays:

- `d_chars`: every character, back to back.
- `d_offsets`: where each string starts, plus one extra entry for the end. The length of string `i` is `offsets[i+1] - offsets[i]`.

```text
Input column: ["CAT", "DOG"]

chars   : [ 'C', 'A', 'T', 'D', 'O', 'G' ]
offsets : [  0 ,   3 ,   6 ]
              ^     ^     ^
           string 0  string 1  end
```

## What's in the code

- **`DeviceBuffer`**: owns a block of GPU memory. It calls `cudaMalloc` on construction and `cudaFree` in the destructor, so memory is released even on early returns or exceptions.
- **No copying, only moving**: the copy constructor is deleted so two buffers can never free the same pointer. The move constructor (`&&`, `noexcept`) hands ownership over by taking the pointer and nulling the original.
- **`StringView`**: a small non-owning struct (24 bytes) that points at the chars and offsets. It is passed by value to the kernel, so threads can read the data without any destructor logic running on the device.
- **`StringsColumn`**: owns the offsets and chars buffers together. Operations like reverse return a new column, so the caller never allocates output memory by hand.

## Project layout

| File | What it does |
| --- | --- |
| `mini_cudf.hpp` | Public header: the memory types and the kernel-launch interface |
| `reverse.cu` | The CUDA kernel and its launch wrapper (built with `nvcc`) |
| `main.cpp` | Plain C++ driver, no CUDA syntax (built with `g++`) |
| `modal_run.py` | Script I use to build and run everything on a T4 via Modal |

The host and device code are compiled separately and linked at the end. `main.cpp` just calls a normal C++ function and never sees CUDA extensions.

```text
main.cpp    --[ g++ ]---> main.o    \
                                     >--[ g++ link ]--> app
reverse.cu  --[ nvcc ]--> reverse.o /
```

## Build and run

```bash
# Device code
nvcc -O3 -c reverse.cu -o reverse.o

# Host code
g++ -O3 -c main.cpp -I/usr/local/cuda/include -o main.o

# Link against the CUDA runtime
g++ main.o reverse.o -L/usr/local/cuda/lib64 -lcudart -o app

./app
```

Expected output:

```text
==================================================
   MILESTONE 4: FULL LIBCUDF-STYLE COLUMN RETURN
==================================================
Input Strings   : "CAT", "DOG"
Output Raw Chars: "TACGOD"

String 0 Reversed: "TAC"
String 1 Reversed: "GOD"
==================================================
Caller allocated ZERO output memory! 100% RAII Owned!
```

## Current limitations

- Only one operation (reverse) is implemented.
- Reversal works on bytes, so it is meant for ASCII text for now.
- I have only checked correctness on small examples, not measured performance.
