#include "core.h"
#include "transport.h"
#include "nvmlwrap.h"
#include "net.h"
#include "param.h"
#include "nvlink.h"
#include <cuda_runtime.h>
#include <assert.h>
#include <mpi.h>
#include <bootstrap.h>
#include "sharp/api/version.h"
#include "sharp/api/sharp_coll.h"

extern void* sharpBootstrapCtx;

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
  MPI_Comm_split(MPI_COMM_WORLD, ring->sharpNodeRank, ring->sharpNodeRank, &(ring->mpiNodeComm));
  //  sharpBootstrapCtx = commState;
  struct sharp_coll_comm_init_spec comm_spec;
  comm_spec.rank      = ring->sharpNodeRank;
  comm_spec.size      = ring->sharpCommSize;
  #if 0
  uint32_t *gwr = NULL;
#if SHARP_API > SHARP_VERSION(1,4)
  gwr = (uint32_t*)malloc(nranks*sizeof(uint32_t));
  gwr[rank] = main_comm->rank;
  NCCLCHECK(bootstrapAllGather(commState, gwr, sizeof(uint32_t)));
  comm_spec.group_world_ranks = gwr;
#endif
  comm_spec.is_comm_world = 0;
  comm_spec.oob_ctx   = commState;
  int ret = sharp_coll_comm_init(main_comm->sharpCtx, &comm_spec, (struct sharp_coll_comm **)&comm->sharpComm);
  if (gwr) free(gwr);
  if (ret < 0) {
      fprintf(stderr, "sharp group create failed:%s(%d)\n", sharp_coll_strerror(ret), ret);
      return ncclInternalError;
  } else {
      fprintf(stderr, "SHARP GROUP CREATE SUCCESS, %p\n", comm->sharpComm);
  }
  #endif
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
  int rank, lrank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_rank(ring->mpiNodeComm, &lrank);
  int offset, sizeReduce;
  offset = sizesFifo[0];
  //sizeReduce = sizesFifo[1];
  sizeReduce = 8;
  
  #if 1
  for(int k = 0; k<4;k++){
    if (rank == k){
      fprintf(stderr, "before allreduce Thread - gRank %d lrank %d: ", rank, lrank);
      for(int l = 0; l < 8; l++)
	fprintf(stderr, "%f ", ((float*)ring->recv.conn.buff)[l+offset]);
      fprintf(stderr, "\n");
    }
    //     MPI_Barrier(MPI_COMM_WORLD);
  }
  #endif
  //  if (rank == 0)
  fprintf(stderr, "grank %d lrank %d: Offset = %d size = %d\n", rank, lrank, offset, sizeReduce);
  MPI_Barrier(MPI_COMM_WORLD);
  //  volatile float* redBuf = (float*)ring->recv.conn.buff;
  float *redBuf = (float*)ring->recv.conn.buff;
  MPI_Allreduce(MPI_IN_PLACE, (float*)redBuf+offset, sizeReduce, MPI_FLOAT, MPI_SUM, ring->mpiNodeComm);
  //  MPI_Barrier(MPI_COMM_WORLD);
  __sync_synchronize();
  #if 0
  for(int k = 0; k<4;k++){
    if (rank == k){
      fprintf(stderr, "after allreduce Thread - grank %d lrank %d: ", rank, lrank);
      for(int l = 0; l < 8; l++)
	fprintf(stderr, "%f ", ((float*)ring->recv.conn.buff)[l+offset]);
      fprintf(stderr, "\n");
    }
    //  MPI_Barrier(MPI_COMM_WORLD);
  }
  #endif
  fprintf(stderr,"Hello world from rank %d!!\n", rank);
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
