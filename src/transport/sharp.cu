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

struct sharpSendResources {
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
  struct ncclSharpContext *sharpSettings;
};

#if 0
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
#endif
ncclResult_t sharpSetup(ncclTinfo_t* myOpaqueInfo, ncclTinfo_t* peerOpaqueInfo, struct ncclConnect* connectInfo, struct ncclRing* ring) {
  struct sharpSendResources* resources;
  NCCLCHECK(ncclCalloc(&resources, 1));
  ring->sharp.transportResources = resources;
  resources->cudaSupport = true;

  int size = offsetof(struct ncclRecvMem, buff)+ring->buffSize;
  if (resources->cudaSupport) {
    NCCLCHECK(ncclCudaCalloc((char**)(&resources->devNetMem), size));
  }
  NCCLCHECK(ncclCudaHostAlloc((void**)&resources->hostRecvMem, (void**)&resources->devHostRecvMem, size));
  NCCLCHECK(ncclCudaHostAlloc((void**)&resources->hostSendMem, (void**)&resources->devHostSendMem, size));
  resources->sharpSettings = ring->sharpSettings;
  //MPI_Comm_split(MPI_COMM_WORLD, ring->mpiColor, ring->mpiColor, &(ring->mpiNodeComm));
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

  //  while (!sizesFifo[2]){
  volatile int* myFlag = ring->sharp.conn.fifo + 2;
  while(*myFlag == 0){
    ;;
  }
  int rank, lrank;
  int offset, count;
  offset = sizesFifo[0];
  count  = sizesFifo[1];
#if 0
#if 1
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_rank(ring->mpiNodeComm, &lrank);
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
  fprintf(stderr, "grank %d lrank %d: Offset = %d size = %d\n", rank, lrank, offset, count);
  MPI_Barrier(MPI_COMM_WORLD);
  float *redBuf = (float*)ring->recv.conn.buff;
  MPI_Allreduce(MPI_IN_PLACE, (float*)redBuf+offset, count, MPI_FLOAT, MPI_SUM, ring->mpiNodeComm);
  MPI_Barrier(MPI_COMM_WORLD);
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
#endif // for MPI allreduce
  //MPI_Barrier(MPI_COMM_WORLD);
  if (count != 0){
    //    float *redBuf = (float*)ring->recv.conn.buff + offset;
    float *redBuf = (float*)resources->sharpSettings->redBuf + offset;
    //  if (resources->sharpSettings->globalRank == 0)
    //   fprintf(stderr, "step = %d start = %f\n", (int)(*prevTail), redBuf[0]);
    struct sharp_coll_reduce_spec *reduce_spec = &resources->sharpSettings->reduce_spec;
    enum sharp_datatype sharp_type;
    enum sharp_reduce_op op_type;
    size_t dt_size;
    sharp_type = SHARP_DTYPE_FLOAT;
    op_type = SHARP_OP_SUM;

    dt_size = sizeof(float);
    reduce_spec->sbuf_desc.buffer.ptr = redBuf;
    void *mr = resources->sharpSettings->mr;
    reduce_spec->sbuf_desc.buffer.length = count * dt_size;
    reduce_spec->sbuf_desc.buffer.mem_handle = mr;
    reduce_spec->sbuf_desc.type = SHARP_DATA_BUFFER;
    reduce_spec->rbuf_desc.buffer.ptr = redBuf;
    reduce_spec->rbuf_desc.buffer.length = count * dt_size;
    reduce_spec->rbuf_desc.buffer.mem_handle = mr;
    reduce_spec->rbuf_desc.type = SHARP_DATA_BUFFER;
    reduce_spec->sbuf_desc.mem_type = SHARP_MEM_TYPE_CUDA;
    reduce_spec->rbuf_desc.mem_type = SHARP_MEM_TYPE_CUDA;
    reduce_spec->length = count;
    reduce_spec->dtype = sharp_type;
    reduce_spec->op = op_type;
    if (SHARP_COLL_SUCCESS != sharp_coll_do_allreduce(resources->sharpSettings->sharpComm, reduce_spec)) {
      WARN("Sharp allreduce failed");
      return ncclInternalError;
    }
  }
    *myFlag = 0;
  __sync_synchronize();
  //++(*prevHead);  
    // ++(*prevTail);  
  return ncclSuccess;
}

extern void* sharpBootstrapCtx;
ncclResult_t sharpConnect(struct ncclConnect* connectInfo, struct ncclConnector* send) {
  // Setup device pointers
  struct sharpSendResources* resources = (struct sharpSendResources*)send->transportResources;
  sharpBootstrapCtx = resources->sharpSettings->commStateNet;
  struct sharp_coll_comm_init_spec comm_spec;
  comm_spec.rank      = resources->sharpSettings->sharpCommRank;
  comm_spec.size      = resources->sharpSettings->sharpCommSize;
  uint32_t *gwr = NULL;
#if SHARP_API > SHARP_VERSION(1,4)
  gwr = (uint32_t*)malloc(resources->sharpSettings->nComms*sizeof(uint32_t));
  gwr[resources->sharpSettings->localRank] = resources->sharpSettings->globalRank;
  NCCLCHECK(bootstrapAllGather(resources->sharpSettings->commStateNet, gwr, sizeof(uint32_t)));
  comm_spec.group_world_ranks = gwr;
#endif 
  comm_spec.is_comm_world = 0;
  comm_spec.oob_ctx   = resources->sharpSettings->commStateNet;
  int ret = sharp_coll_comm_init(resources->sharpSettings->sharpCtx, &comm_spec, (struct sharp_coll_comm **)&resources->sharpSettings->sharpComm);
  //->sharpCtx = main_comm->sharpSettings.sharpCtx;
  if (gwr) free(gwr);
  if (ret < 0) {
    WARN("Sharp group create failed: %s(%d)", sharp_coll_strerror(ret), ret);
    return ncclInternalError;
  }
  if (SHARP_COLL_SUCCESS != sharp_coll_reg_mr(resources->sharpSettings->sharpCtx, resources->sharpSettings->redBuf, resources->sharpSettings->redBufSize, &(resources->sharpSettings->mr))){
    WARN("Sharp reg mr failed");
    return ncclInternalError;
  }
  
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
  return ncclSuccess;
}

ncclResult_t sharpFree(void* transportResources) {
  fprintf(stderr,"sharp free\n");
  struct sharpSendResources* resources = (struct sharpSendResources*)transportResources;
  NCCLCHECK(ncclCudaHostFree(resources->hostSendMem));
  NCCLCHECK(ncclCudaHostFree(resources->hostRecvMem));
  if (resources->cudaSupport)
    CUDACHECK(cudaFree(resources->devNetMem));
  if (SHARP_COLL_SUCCESS != sharp_coll_dereg_mr(resources->sharpSettings->sharpCtx, resources->sharpSettings->mr)) {
    WARN("Sharp dereg mr failed");
    return ncclInternalError;
  }
  free(resources);
  return ncclSuccess;
}
struct ncclTransport sharpTransport = {
				       "SRP",
				       NULL,
				       NULL,
				       NULL,
				       {sharpSetup, sharpConnect, sharpFree, sharpProxy},
				       {sharpSetup, sharpConnect, sharpFree, sharpProxy}
};
