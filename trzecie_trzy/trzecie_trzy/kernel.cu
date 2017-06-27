#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <cstdlib>
#include <cmath>

#define PROMIEN       1
#define BLOCK_SIZE    6
#define NUM_ELEMENTS  (5) //WYNIKOW
//przygotowac kod dla dowolnych BLOCK_SIZE, NUM_ELEMENTS
// PROMIEN<BLOCK_SIZE

// CUDA API MAKRO SPRAWDZANIA BLEDOW

#define cudaCheck(error) \
if (error != cudaSuccess) { \
	printf("BLAD URUCHOMINIA: %s at %s:%d\n", cudaGetErrorString(error), __FILE__, __LINE__); \
	exit(1); \
}

//KERNEL 
__global__ void wzorzec_1w(int *in, int *out, int size)
{
	__shared__ int temp_in[BLOCK_SIZE + 2 * PROMIEN];

	//ELEMENT ŒRODKOWY DLA W¥TKU (ADRES GLOBALNY)
	int gindex = threadIdx.x + (blockIdx.x * blockDim.x) + PROMIEN;
	//ELEMENT ŒRODKOWY DLA W¥TKU (ADRES LOKALNY)
	int lindex = threadIdx.x + PROMIEN;

	int result = 0;
	for (int i = -PROMIEN; i <= PROMIEN; i++)
	{
		result += in[lindex + i];
	}
	out[gindex - PROMIEN] = result;

}

int main()
{
	unsigned int i;
	int h_in[NUM_ELEMENTS + 2 * PROMIEN], h_out[NUM_ELEMENTS];
	int *d_in, *d_out;
	char msg;
	// DANE WEJSCIOWE - INICJALIZACJA
	for (i = 0; i < (NUM_ELEMENTS + 2 * PROMIEN); ++i)
		h_in[i] = i;
	// PRZYK£AD: Dla 1 i PROMIEN = n, wszystkie wyniki powinny 
	// byæ równe 2n+1

	printf("Wejsciowe \n");
	for (int i = 0; i < NUM_ELEMENTS + 2 * PROMIEN; i++)
		printf("%d ", h_in[i]);


	// pamiêæ globalna karty 
	cudaCheck(cudaMalloc(&d_in, (NUM_ELEMENTS + 2 * PROMIEN) * sizeof(int)));
	cudaCheck(cudaMalloc(&d_out, NUM_ELEMENTS * sizeof(int)));

	// kopiowanie wejœcia
	cudaCheck(cudaMemcpy(d_in, h_in, (NUM_ELEMENTS + 2 * PROMIEN) * sizeof(int), cudaMemcpyHostToDevice));
	cudaError_t err = cudaGetLastError();
	//URUCHMIENIE PRZETARZANIA 
	//liczba blokow: do zrobienia
	int grid = (int)ceil((1.0*(NUM_ELEMENTS + 2 * PROMIEN)) / (1.0*BLOCK_SIZE));
	//printf("%d ", grid);
	wzorzec_1w <<< 1, BLOCK_SIZE >> > (d_in, d_out, NUM_ELEMENTS);



	cudaThreadSynchronize();
	err = cudaGetLastError();
	/*if (cudaSuccess != err)
	{
	fprintf(stderr, "Cuda blad: %s: %s.\n", msg, cudaGetErrorString(err));
	//exit(EXIT_FAILURE);
	}*/


	//kopiowanie wyjscia 
	cudaCheck(cudaMemcpy(h_out, d_out, NUM_ELEMENTS * sizeof(int), cudaMemcpyDeviceToHost));

	printf("\nWyjsciowe \n");
	for (int i = 0; i < NUM_ELEMENTS; i++){
		printf("%d ", h_out[i]);
		if (i == NUM_ELEMENTS - 1)
			printf("GRATULACJE obliczenia poprawne!\n");
	}
		
	printf("\n");

	printf("de1");

	printf("de2");
	// Porzadki
	cudaFree(d_in);
	cudaFree(d_out);
	cudaCheck(cudaDeviceReset());
	printf("de3");
	getchar();
	
	return 0;
}
