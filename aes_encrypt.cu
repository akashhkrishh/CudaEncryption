#include <iostream>
#include <fstream>
#include <cstdlib>
#include <cuda_runtime.h>

using namespace std;

#define BLOCK_SIZE 16 // AES block size (128 bits)

// A simplified AES encryption step (just for demonstration; NOT actual AES)
__device__ void aes_encrypt_block(unsigned char* block, unsigned char* key) {
    for (int i = 0; i < BLOCK_SIZE; i++) {
        block[i] ^= key[i];  // Just an example, XOR with key (not AES encryption)
    }
}

// CUDA kernel for AES encryption
__global__ void AES_EncryptKernel(unsigned char* data, unsigned char* key, int numBlocks) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numBlocks) {
        unsigned char* block = &data[idx * BLOCK_SIZE];
        aes_encrypt_block(block, key);
    }
}

void encryptFile(const char* inputFile, const char* outputFile, const unsigned char* key) {
    ifstream ifs(inputFile, ios::binary | ios::ate);
    if (!ifs) {
        cerr << "Cannot open input file!" << endl;
        exit(1);
    }

    streampos fileSize = ifs.tellg();
    ifs.seekg(0, ios::beg);

    // Cast to size_t to avoid errors when calculating numBlocks
    size_t numBlocks = (static_cast<size_t>(fileSize) + BLOCK_SIZE - 1) / BLOCK_SIZE; // Round up number of blocks
    unsigned char* data = new unsigned char[numBlocks * BLOCK_SIZE];
    ifs.read(reinterpret_cast<char*>(data), fileSize);
    ifs.close();

    // Allocate device memory for data and key
    unsigned char* d_data;
    unsigned char* d_key;
    cudaMalloc(&d_data, numBlocks * BLOCK_SIZE);
    cudaMalloc(&d_key, BLOCK_SIZE);

    // Copy data to device memory
    cudaMemcpy(d_data, data, numBlocks * BLOCK_SIZE, cudaMemcpyHostToDevice);
    cudaMemcpy(d_key, key, BLOCK_SIZE, cudaMemcpyHostToDevice);

    // Launch the encryption kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (numBlocks + threadsPerBlock - 1) / threadsPerBlock;
    AES_EncryptKernel<<<blocksPerGrid, threadsPerBlock>>>(d_data, d_key, numBlocks);

    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        cerr << "CUDA kernel launch failed: " << cudaGetErrorString(err) << endl;
        exit(1);
    }

    // Wait for GPU to finish
    cudaDeviceSynchronize();

    // Copy the result back to host memory
    cudaMemcpy(data, d_data, numBlocks * BLOCK_SIZE, cudaMemcpyDeviceToHost);

    // Write the encrypted data to the output file
    ofstream ofs(outputFile, ios::binary);
    ofs.write(reinterpret_cast<char*>(data), numBlocks * BLOCK_SIZE);
    ofs.close();

    // Free device memory
    cudaFree(d_data);
    cudaFree(d_key);

    // Free host memory
    delete[] data;
}

int main() {
    // AES key (128 bits)
    unsigned char key[16] = {0x2b, 0x7e, 0x15, 0x16, 0x28, 0xae, 0xd2, 0xa6,
                             0xab, 0xf7, 0x97, 0x75, 0x46, 0x38, 0x6d, 0x60};

    const char* inputFile = "./novel.txt";   // Path to your input file
    const char* encryptedFile = "./encrypted.bin"; // Output encrypted file

    // Encrypt the file
    encryptFile(inputFile, encryptedFile, key);
    cout << "File encrypted successfully!" << endl;

    return 0;
}
