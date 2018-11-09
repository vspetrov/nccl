#include "core.h"
#include "transport.h"
#include "nvmlwrap.h"
#include "net.h"
#include "param.h"
#include "nvlink.h"
#include <cuda_runtime.h>
#include <assert.h>
#include <mpi.h>

struct sharpSendResources {
  void* netSendComm;
  struct ncclSendMem* hostSendMem;
  struct ncclRecvMem* hostRecvMem;
  struct ncclSendMem* devHostSendMem;
  struct ncclRecvMem* devHostRecvMem;
  struct ncclSendMem* hostDevMem;
  int netDev;
  bool cudaSupport;
  struct ncclRecvMem* devNetMem;
  uint64_t llStep;
  uint64_t llLastCleaning;
};

struct sharpRecvResources {
  void* netListenComm;
  void* netRecvComm;
  struct ncclSendMem* hostSendMem;
  struct ncclRecvMem* hostRecvMem;
  struct ncclSendMem* devHostSendMem;
  struct ncclRecvMem* devHostRecvMem;
  struct ncclRecvMem* hostDevMem;
  int netDev;
  bool cudaSupport;
  uint64_t llStep;
  uint64_t llLastCleaning;
};

ncclResult_t doSharp (struct ncclProxyArgs* args){
  fprintf(stderr,"Running sharp thread\n");
  return ncclSuccess;
  
}

ncclResult_t sharpSetup(ncclTinfo_t* myOpaqueInfo, ncclTinfo_t* peerOpaqueInfo, struct ncclConnect* connectInfo, struct ncclRing* ring) {
  struct sharpSendResources* resources;
  NCCLCHECK(ncclCalloc(&resources, 1));
  ring->sharp.transportResources = resources;

  fprintf(stderr,"Sharp setup\n");
  #if 0
  struct netInfo* myInfo = (struct netInfo*)myOpaqueInfo;
  resources->netDev = getDev(ring->id, myInfo->ndev, myInfo->scores);
  resources->cudaSupport = false;
#endif

  int cudaDev;
  CUDACHECK(cudaGetDevice(&cudaDev));
  resources->cudaSupport = true;

  int size = offsetof(struct ncclRecvMem, buff)+ring->buffSize;
  if (resources->cudaSupport) {
    NCCLCHECK(ncclCudaCalloc((char**)(&resources->devNetMem), size));
  }

  NCCLCHECK(ncclCudaHostAlloc((void**)&resources->hostRecvMem, (void**)&resources->devHostRecvMem, size));
  NCCLCHECK(ncclCudaHostAlloc((void**)&resources->hostSendMem, (void**)&resources->devHostSendMem, size));

  return ncclSuccess;
}


ncclResult_t sharpProxy(struct ncclProxyArgs* args) {
  struct ncclRing* ring = args->ring;
  struct sharpSendResources* resources = (struct sharpSendResources*) (ring->sharp.transportResources);
  const int llMode = args->llMode;
  volatile uint64_t* prevTail = &resources->hostRecvMem->tail;

  struct ncclSendMem* prevMem = resources->hostDevMem ? resources->hostDevMem : resources->hostSendMem;
  volatile uint64_t* prevHead = llMode ? &prevMem->llHead : &prevMem->head;
  struct ncclRecvMem* localMem = resources->cudaSupport ? resources->devNetMem : resources->hostRecvMem;
  char* localBuff = llMode ? resources->hostRecvMem->llBuff : localMem->buff;
  volatile int* sizesFifo = llMode ? resources->hostRecvMem->llSizesFifo : resources->hostRecvMem->sizesFifo;
  int buffSize = llMode ? NCCL_LL_BUFF_SIZE : ring->buffSize;
  int sliceSize = buffSize / args->substeps;

  while (!(*prevHead)){ 
    ;;
  }
  MPI_Comm sepComm;
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  int color = rank % 2;
  MPI_Comm_split(MPI_COMM_WORLD, color, rank, &sepComm);
  for(int k = 0; k<4;k++){
    if (rank == k){
      fprintf(stderr, "before allreduce Thread - Rank %d: ", rank);
      for(int l = 0; l < 8; l++)
	fprintf(stderr, "%f ", ((float*)ring->recv.conn.buff)[l+524288]);
      fprintf(stderr, "\n");
    }
      MPI_Barrier(MPI_COMM_WORLD);
  }
  
  MPI_Allreduce(MPI_IN_PLACE, ((float*)ring->recv.conn.buff)+524288, 8, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
  
  for(int k = 0; k<4;k++){
    if (rank == k){
      fprintf(stderr, "after allreduce Thread - Rank %d: ", rank);
      for(int l = 0; l < 8; l++)
	fprintf(stderr, "%f ", ((float*)ring->recv.conn.buff)[l]);
      fprintf(stderr, "\n");
    }
      MPI_Barrier(MPI_COMM_WORLD);
  }
  
  MPI_Comm_free(&sepComm);
  ++(*prevTail);  
  return ncclSuccess;
}

ncclResult_t sharpConnect(struct ncclConnect* connectInfo, struct ncclConnector* send) {
  // Setup device pointers
  struct sharpSendResources* resources = (struct sharpSendResources*)send->transportResources;

  if (resources->cudaSupport) {
    send->conn.buff = resources->devNetMem->buff;
    // We don't use devMem for llMode because the CPU has to read the data
    send->conn.llBuff = resources->devHostRecvMem->llBuff;
  } else {
    send->conn.buff = resources->devHostRecvMem->buff;
    send->conn.llBuff = resources->devHostRecvMem->llBuff;
  }
  send->conn.tail = &resources->devHostRecvMem->tail;
  send->conn.opCount = &resources->devHostRecvMem->opCount;
  send->conn.fifo = resources->devHostRecvMem->sizesFifo;
  send->conn.llFifo = resources->devHostRecvMem->llSizesFifo;

  if (resources->hostDevMem == NULL) {
    send->conn.head = &resources->devHostSendMem->head;
    send->conn.llHead = &resources->devHostSendMem->llHead;
  }

  //Sharp comm init should be here.
  return ncclSuccess;
}

struct ncclTransport sharpTransport = {
				       "SRP",
				       NULL,
				       NULL,
				       NULL,
				       {sharpSetup, sharpConnect, NULL, sharpProxy},
				       {sharpSetup, sharpConnect, NULL, sharpProxy}
};
