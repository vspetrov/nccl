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
#include <sys/ipc.h> 
#include <sys/shm.h> 


struct sharpIPC{
  cudaIpcMemHandle_t devIpc;
  void *remPtr;
};

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
  void *sharpBuf;
  void *devSharpBuf;
  struct ncclSharpContext *sharpSettings;
  struct sharpIPC *nodeRanksBuf;
  int *flags;
};

ncclResult_t sharpSetup(ncclTinfo_t* myOpaqueInfo, ncclTinfo_t* peerOpaqueInfo, struct ncclConnect* connectInfo, struct ncclRing* ring) {
  struct sharpSendResources* resources;
  NCCLCHECK(ncclCalloc(&resources, 1));
  ring->sharp.transportResources = resources;
  resources->cudaSupport = true;

  int size = offsetof(struct ncclRecvMem, buff)+ring->buffSize;
  if (resources->cudaSupport) {
    NCCLCHECK(ncclCudaCalloc((char**)(&resources->devNetMem), size));
  }
   
  //NCCLCHECK(ncclCudaHostAlloc((void**)&resources->sharpBuf, (void**)&resources->devSharpBuf, NCCL_LL_BUFF_SIZE));
  NCCLCHECK(ncclCudaCalloc((char**)&resources->sharpBuf, ring->buffSize + 1));
  resources->devSharpBuf = resources->sharpBuf;
  NCCLCHECK(ncclCudaHostAlloc((void**)&resources->hostRecvMem, (void**)&resources->devHostRecvMem, size));
  NCCLCHECK(ncclCudaHostAlloc((void**)&resources->hostSendMem, (void**)&resources->devHostSendMem, size));
  resources->sharpSettings = ring->sharpSettings;
  return ncclSuccess;
}

static void create_iov_buffer(struct sharp_coll_data_desc *desc, int iov_count,  sharpSendResources *resources, int length)
{
	desc->type = SHARP_DATA_IOV;
	desc->iov.count = iov_count;
	desc->iov.vector = (sharp_data_iov*)malloc (iov_count * sizeof(struct sharp_data_iov));
	assert(desc->iov.vector != NULL);

	desc->iov.vector[0].ptr = resources->sharpSettings->redBuf;
	desc->iov.vector[0].mem_handle = resources->sharpSettings->mrs[0];
	desc->iov.vector[0].length = length;
#if 0
	for (int i = 1; i < iov_count; i++) {
		desc->iov.vector[i].ptr = resources->nodeRanksBuf[i].remPtr;
		desc->iov.vector[i].mem_handle = resources->sharpSettings->mrs[i];
		desc->iov.vector[i].length = length;
	}
#endif
}

ncclResult_t sharpProxy(struct ncclProxyArgs* args) {
  struct ncclRing* ring = args->ring;
  struct sharpSendResources* resources = (struct sharpSendResources*) (ring->sharp.transportResources);
  const int llMode = args->llMode;

  volatile int* sizesFifo = llMode ? resources->hostRecvMem->llSizesFifo : resources->hostRecvMem->sizesFifo;
  int buffSize = llMode ? NCCL_LL_BUFF_SIZE : ring->buffSize;
  int iter = 0;
  while(1){
  volatile int* myFlag =  sizesFifo;
  while(sizesFifo[2] == 0){
    ;;
  }

  int offset, count;
  
  offset = llMode? sizesFifo[0]:sizesFifo[0];
  count  = llMode? sizesFifo[1]:sizesFifo[1];
  resources->flags[resources->sharpSettings->localRank] = count;
  if (resources->sharpSettings->globalRank == resources->sharpSettings->nodeLeaderRank){
    // if (count >= 0){

    for(int r = 1; r < resources->sharpSettings->nodeCommSize; r++){
      volatile int* rankFlag = &(resources->flags[r]);
      int isReady = *rankFlag;
      while(!isReady){
	isReady = *rankFlag;
      }
    }
#if 0 //sharp v1
    float *redBuf;
    if (llMode){
      redBuf = (float*)resources->sharpSettings->llRedBuf + offset;
    }
    else{
      redBuf = (float*)resources->sharpSettings->redBuf + offset;
    }
#endif
    //fprintf(stderr, "Start sharp rank=%d count=%d  iter=%d\n", resources->sharpSettings->globalRank, count, iter );
    // iter++;
    //  if (resources->sharpSettings->globalRank == 0)
    //   fprintf(stderr, "step = %d start = %f\n", (int)(*prevTail), redBuf[0]);
    //    fprintf(stderr, "rank = %d, size=%d\n", resources->sharpSettings->globalRank, count);
    struct sharp_coll_reduce_spec *reduce_spec = &resources->sharpSettings->reduce_spec;
    enum sharp_datatype sharp_type;
    enum sharp_reduce_op op_type;
    size_t dt_size;
    sharp_type = SHARP_DTYPE_FLOAT;
    op_type = SHARP_OP_SUM;

    dt_size = sizeof(float);
    void *mr = llMode ? resources->sharpSettings->llmr : resources->sharpSettings->mr;

    create_iov_buffer(&reduce_spec->sbuf_desc, resources->sharpSettings->nodeCommSize,  resources, count * dt_size);
    create_iov_buffer(&reduce_spec->rbuf_desc, resources->sharpSettings->nodeCommSize,  resources, count * dt_size);
    #if 0
    reduce_spec->sbuf_desc.buffer.ptr = redBuf;    
    reduce_spec->sbuf_desc.buffer.length = count * dt_size;
    reduce_spec->sbuf_desc.buffer.mem_handle = mr;
    reduce_spec->sbuf_desc.type = SHARP_DATA_BUFFER;
    reduce_spec->sbuf_desc.mem_type = SHARP_MEM_TYPE_CUDA;
    
    reduce_spec->rbuf_desc.buffer.ptr = redBuf;
    reduce_spec->rbuf_desc.buffer.length = count * dt_size;
    reduce_spec->rbuf_desc.buffer.mem_handle = mr;
    reduce_spec->rbuf_desc.type = SHARP_DATA_BUFFER;
    reduce_spec->rbuf_desc.mem_type = SHARP_MEM_TYPE_CUDA;
    #endif
    reduce_spec->length = count;
    reduce_spec->dtype = sharp_type;
    reduce_spec->op = op_type;
    #if 1
    if (SHARP_COLL_SUCCESS != sharp_coll_do_allreduce(resources->sharpSettings->sharpComm, reduce_spec)) {
      WARN("Sharp allreduce failed");
      return ncclInternalError;
    }
    #endif
    for(int r = 1; r < resources->sharpSettings->nodeCommSize; r++){
      volatile int* rankFlag = &(resources->flags[r]);
      *rankFlag = 0;
    }
  }
  else{
      volatile int* rankFlag = &(resources->flags[resources->sharpSettings->localRank]);
      int isReady = *rankFlag;
      while(isReady != 0){
	isReady = *rankFlag;
      }    
  }
  
  
  //    fprintf(stderr, "finished sharp\n");
  volatile int* flag2 = sizesFifo +2;
  bool lastForThisColl = (*flag2==-1);
  //    sizesFifo[2] = 0;
  *flag2 = 0;
  __sync_synchronize();
  if (lastForThisColl)
    break;
  //++(*prevHead);  
    // ++(*prevTail);
  }
  return ncclSuccess;
}

extern void* sharpBootstrapCtx;
ncclResult_t sharpConnect(struct ncclConnect* connectInfo, struct ncclConnector* send) {
  // Setup device pointers
  struct sharpSendResources* resources = (struct sharpSendResources*)send->transportResources;

  send->conn.tail = &resources->devHostRecvMem->tail;
  send->conn.opCount = &resources->devHostRecvMem->opCount;
  send->conn.fifo = resources->devHostRecvMem->sizesFifo;
  send->conn.llFifo = resources->devHostRecvMem->llSizesFifo;

  //resources->hostRecvMem->llSizesFifo[2] = 0;
  send->conn.fifo[2] = 0;
  send->conn.llFifo[2] = 0;

  //buffers accessed in kernel
  send->conn.llBuff = (char*)resources->devSharpBuf;
  send->conn.buff  =  (char*)resources->devSharpBuf;
  
  if (resources->hostDevMem == NULL) {
    send->conn.head = &resources->devHostSendMem->head;
    send->conn.llHead = &resources->devHostSendMem->llHead;
  }
  resources->sharpSettings->llRedBuf = resources->sharpBuf;
  resources->sharpSettings->redBuf   = resources->sharpBuf;

  key_t key = ftok("ncclshmem", 101);
  int shmid = shmget(key, sizeof(int) * resources->sharpSettings->nodeCommSize, 0666|IPC_CREAT);
  resources->flags = (int *)shmat(shmid,(void*)0,0);
  resources->flags[resources->sharpSettings->localRank] = 0;
  NCCLCHECK(ncclCalloc(&(resources->nodeRanksBuf), resources->sharpSettings->nodeCommSize));

  cudaError_t err = cudaIpcGetMemHandle(&resources->nodeRanksBuf[resources->sharpSettings->localRank].devIpc, (void*)resources->sharpBuf);
  if (cudaSuccess != err){
    WARN("Failed to get CUDA IPC handle during sharp setup");
    return ncclInternalError;
  }
  NCCLCHECK(bootstrapAllGather(resources->sharpSettings->oobNodeContext, resources->nodeRanksBuf, sizeof(struct sharpIPC)));
  if (resources->sharpSettings->globalRank == resources->sharpSettings->nodeLeaderRank){
    for(int r = 1; r < resources->sharpSettings->nodeCommSize; r++){
      cudaError_t err = cudaIpcOpenMemHandle(&resources->nodeRanksBuf[r].remPtr, resources->nodeRanksBuf[r].devIpc, cudaIpcMemLazyEnablePeerAccess);
      if (cudaSuccess != err){
	WARN("Failed to open CUDA IPC handle for rank %d, %s", 1, cudaGetErrorString(err));
	return ncclInternalError;
      }
      else{
	fprintf(stderr, "Successfully opened ipc handler for rank %d\n", r);
      }
    }
    NCCLCHECK(ncclCalloc(&(resources->sharpSettings->mrs), resources->sharpSettings->nodeCommSize));
  }
  
  //setup sharp
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
  #if 1
  if (resources->sharpSettings->globalRank == resources->sharpSettings->nodeLeaderRank){
    if (SHARP_COLL_SUCCESS != sharp_coll_reg_mr(resources->sharpSettings->sharpCtx, resources->sharpSettings->redBuf, resources->sharpSettings->redBufSize, &(resources->sharpSettings->mrs[0]))){
      WARN("Sharp reg mr failed for reduction buffer");
      return ncclInternalError;
    }
    #if 1
    for(int r = 1; r < resources->sharpSettings->nodeCommSize; r++){
      if (SHARP_COLL_SUCCESS != sharp_coll_reg_mr(resources->sharpSettings->sharpCtx, resources->nodeRanksBuf[r].remPtr, resources->sharpSettings->redBufSize, &(resources->sharpSettings->mrs[r]))){
	WARN("Sharp reg mr failed for reduction buffer");
	return ncclInternalError;  
      }
    }
    #endif
  }
  #endif
  #if 0
  if (SHARP_COLL_SUCCESS != sharp_coll_reg_mr(resources->sharpSettings->sharpCtx, resources->sharpSettings->llRedBuf, NCCL_LL_BUFF_SIZE, &(resources->sharpSettings->llmr))){
    WARN("Sharp reg mr failed for reduction buffer");
    return ncclInternalError;
  }
  #endif

  return ncclSuccess;
}

ncclResult_t sharpFree(void* transportResources) {
  fprintf(stderr,"sharp free\n");
  struct sharpSendResources* resources = (struct sharpSendResources*)transportResources;
  NCCLCHECK(ncclCudaHostFree(resources->hostSendMem));
  NCCLCHECK(ncclCudaHostFree(resources->hostRecvMem));
  if (resources->cudaSupport)
    CUDACHECK(cudaFree(resources->devNetMem));
  shmdt(resources->flags);
  if (resources->sharpSettings->globalRank == resources->sharpSettings->nodeLeaderRank){
    for(int r = 0; r < resources->sharpSettings->nodeCommSize; r++){
      if (SHARP_COLL_SUCCESS != sharp_coll_dereg_mr(resources->sharpSettings->sharpCtx, resources->sharpSettings->mrs[r])) {
	WARN("Sharp dereg mr failed");
	return ncclInternalError;
      }
    }
    for(int r = 1; r < resources->sharpSettings->nodeCommSize; r++){
      CUDACHECK(cudaIpcCloseMemHandle(resources->nodeRanksBuf[r].remPtr));
    }
  }
  free(resources->nodeRanksBuf);
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
