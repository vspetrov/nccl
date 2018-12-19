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
  // NCCLCHECK(ncclCudaCalloc((char**)&resources->sharpBuf, ring->buffSize + 1));
  //resources->devSharpBuf = resources->sharpBuf;
  NCCLCHECK(ncclCudaHostAlloc((void**)&resources->hostRecvMem, (void**)&resources->devHostRecvMem, size));
  NCCLCHECK(ncclCudaHostAlloc((void**)&resources->hostSendMem, (void**)&resources->devHostSendMem, size));
  resources->sharpSettings = ring->sharpSettings;
  return ncclSuccess;
}

static void create_iov_buffer(struct sharp_coll_data_desc *desc, int iov_count,  sharpSendResources *resources){
  int *length;
  length = resources->flags;
  desc->type = SHARP_DATA_IOV;
  desc->iov.count = iov_count;
  desc->iov.vector = (sharp_data_iov*)malloc (iov_count * sizeof(struct sharp_data_iov));
  assert(desc->iov.vector != NULL);

  for (int i = 0; i < iov_count; i++) {
    desc->iov.vector[i].ptr = resources->nodeRanksBuf[i].remPtr;
    desc->iov.vector[i].mem_handle = resources->sharpSettings->mrs[i];
    desc->iov.vector[i].length = length[2*i + 1] * sizeof(float);
  }
}
static void initIOVbuffer(struct sharp_coll_data_desc *desc, int iov_count,  sharpSendResources *resources){
  desc->type = SHARP_DATA_IOV;
  desc->iov.count = iov_count;
  desc->iov.vector = (sharp_data_iov*)malloc (iov_count * sizeof(struct sharp_data_iov));
  assert(desc->iov.vector != NULL);
  for (int i = 0; i < iov_count; i++) {
    desc->iov.vector[i].ptr = resources->nodeRanksBuf[i].remPtr;
    desc->iov.vector[i].mem_handle = resources->sharpSettings->mrs[i];
  }
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
  resources->flags[2*resources->sharpSettings->localRank] = 1;
  resources->flags[2*resources->sharpSettings->localRank + 1] = count;
  if (resources->sharpSettings->globalRank == resources->sharpSettings->nodeLeaderRank){
    // if (count >= 0){
    for(int r = 1; r < resources->sharpSettings->nodeCommSize; r++){
      volatile int* rankFlag = &(resources->flags[2*r]);
      int isReady = *rankFlag;
      while(!isReady){
	isReady = *rankFlag;
      }
    }
    #if 0
    for(int r = 0; r < resources->sharpSettings->nodeCommSize; r++){
      fprintf(stderr, "Start sharp grank=%d lrank=%d count=%d  iter=%d\n", resources->sharpSettings->globalRank, r, resources->flags[2*r+1], iter );
    }
    #endif
    iter++;
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
    //create_iov_buffer(&reduce_spec->sbuf_desc, resources->sharpSettings->nodeCommSize,  resources);
    //create_iov_buffer(&reduce_spec->rbuf_desc, resources->sharpSettings->nodeCommSize,  resources);
    int totalCount = 0;
    for(int r = 0; r < resources->sharpSettings->nodeCommSize; r++){
      reduce_spec->sbuf_desc.iov.vector[r].length = resources->flags[2*r + 1] * sizeof(float);
      reduce_spec->rbuf_desc.iov.vector[r].length = resources->flags[2*r + 1] * sizeof(float);
      totalCount += resources->flags[2*r + 1];
    }
    reduce_spec->length = totalCount;
    reduce_spec->dtype = sharp_type;
    reduce_spec->op = op_type;
    #if 1
    if (SHARP_COLL_SUCCESS != sharp_coll_do_allreduce(resources->sharpSettings->sharpComm, reduce_spec)) {
      WARN("Sharp allreduce failed");
      return ncclInternalError;
    }
    #endif
    for(int r = 1; r < resources->sharpSettings->nodeCommSize; r++){
      volatile int* rankFlag = &(resources->flags[2*r]);
      *rankFlag = 0;
    }
  }
  else{
      volatile int* rankFlag = &(resources->flags[2*resources->sharpSettings->localRank]);
      int isReady = *rankFlag;
      while(isReady != 0){
	isReady = *rankFlag;
      }    
  }
  volatile int* flag2 = sizesFifo +2;
  bool lastForThisColl = (*flag2==-1);
  *flag2 = 0;
  __sync_synchronize();
  if (lastForThisColl)
    break;
  }
  return ncclSuccess;
}

extern void* sharpBootstrapCtx;
ncclResult_t sharpConnect(struct ncclConnect* connectInfo, struct ncclConnector* send) {
  // Setup device pointers
  struct sharpSendResources* resources = (struct sharpSendResources*)send->transportResources;
  int my_device;
  
  send->conn.tail = &resources->devHostRecvMem->tail;
  send->conn.opCount = &resources->devHostRecvMem->opCount;
  send->conn.fifo = resources->devHostRecvMem->sizesFifo;
  send->conn.llFifo = resources->devHostRecvMem->llSizesFifo;

  //resources->hostRecvMem->llSizesFifo[2] = 0;
  send->conn.fifo[2] = 0;
  send->conn.llFifo[2] = 0;

  if (resources->hostDevMem == NULL) {
    send->conn.head = &resources->devHostSendMem->head;
    send->conn.llHead = &resources->devHostSendMem->llHead;
  }

  key_t key;
  if ((key = ftok("/root/tmp.nccl", 101)) == -1) {
    WARN("ftok failed %s\n",strerror(errno));
  }
  int shmid;
  if ((shmid = shmget(key, 2 * sizeof(int) * resources->sharpSettings->nodeCommSize, 0666|IPC_CREAT)) == -1){
    WARN("shmget failed %s\n", strerror(errno));
  }
  if ((resources->flags = (int *)shmat(shmid,(void*)0,0)) == (void*)-1){
    WARN("shmid failed %s\n", strerror(errno));
  }
  resources->flags[2*resources->sharpSettings->localRank] = 0;
  resources->flags[2*resources->sharpSettings->localRank + 1] = 0;
  NCCLCHECK(ncclCalloc(&(resources->nodeRanksBuf), resources->sharpSettings->nodeCommSize));
  if (cudaGetDevice(&my_device) != cudaSuccess) {
    WARN("Failed to get my device");
  }

  if (resources->sharpSettings->globalRank == resources->sharpSettings->nodeLeaderRank){
      for(int r = 1; r < resources->sharpSettings->nodeCommSize; r++){
	if(cudaSetDevice(r) != cudaSuccess) {
	  WARN("Failed to set cuda deice to :%d",r);
	}
      NCCLCHECK(ncclCudaCalloc((char**)&resources->nodeRanksBuf[r].remPtr, resources->sharpSettings->redBufSize));
      cudaError_t err = cudaIpcGetMemHandle(&resources->nodeRanksBuf[r].devIpc, (void*)resources->nodeRanksBuf[r].remPtr);
      if (cudaSuccess != err){
	WARN("Failed to get CUDA IPC handle during sharp setup");
	return ncclInternalError;
      }
    }
      if (cudaSetDevice(my_device) != cudaSuccess) {
	WARN("Failed to set my current device :%d", my_device);
      }
    NCCLCHECK(ncclCudaCalloc((char**)&resources->nodeRanksBuf[0].remPtr, resources->sharpSettings->redBufSize));
    resources->sharpBuf =  resources->nodeRanksBuf[0].remPtr;
    resources->devSharpBuf =  resources->nodeRanksBuf[0].remPtr;
  }
  bootstrapBcast(resources->sharpSettings->oobNodeContext, resources->nodeRanksBuf, resources->sharpSettings->nodeCommSize * sizeof(struct sharpIPC), 0);
  if (resources->sharpSettings->globalRank == resources->sharpSettings->nodeLeaderRank){
    NCCLCHECK(ncclCalloc(&(resources->sharpSettings->mrs), resources->sharpSettings->nodeCommSize));
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
  }
  else {
    cudaError_t err = cudaIpcOpenMemHandle(&resources->nodeRanksBuf[resources->sharpSettings->localRank].remPtr, resources->nodeRanksBuf[resources->sharpSettings->localRank].devIpc, cudaIpcMemLazyEnablePeerAccess);
    if (cudaSuccess != err){
      WARN("Failed to open CUDA IPC handle for rank %d, %s", 1, cudaGetErrorString(err));
      return ncclInternalError;
    }
    resources->sharpBuf    = resources->nodeRanksBuf[resources->sharpSettings->localRank].remPtr;
    resources->devSharpBuf = resources->sharpBuf;
  }
  resources->sharpSettings->llRedBuf = resources->sharpBuf;
  resources->sharpSettings->redBuf   = resources->sharpBuf;
  send->conn.llBuff = (char*)resources->devSharpBuf;
  send->conn.buff  =  (char*)resources->devSharpBuf;
  if (resources->sharpSettings->globalRank == resources->sharpSettings->nodeLeaderRank){
    for(int r = 0; r < resources->sharpSettings->nodeCommSize; r++){
      if (SHARP_COLL_SUCCESS != sharp_coll_reg_mr(resources->sharpSettings->sharpCtx, resources->nodeRanksBuf[r].remPtr, resources->sharpSettings->redBufSize, &(resources->sharpSettings->mrs[r]))){
	WARN("Sharp reg mr failed for reduction buffer");
	return ncclInternalError;  
      }
    }
    struct sharp_coll_reduce_spec *reduce_spec = &resources->sharpSettings->reduce_spec;
    initIOVbuffer(&reduce_spec->sbuf_desc, resources->sharpSettings->nodeCommSize,  resources);
    initIOVbuffer(&reduce_spec->rbuf_desc, resources->sharpSettings->nodeCommSize,  resources);
  }
  return ncclSuccess;
}

ncclResult_t sharpFree(void* transportResources) {
  struct sharpSendResources* resources = (struct sharpSendResources*)transportResources;
  NCCLCHECK(ncclCudaHostFree(resources->hostSendMem));
  NCCLCHECK(ncclCudaHostFree(resources->hostRecvMem));

  if (resources->cudaSupport)
    CUDACHECK(cudaFree(resources->devNetMem));
  shmdt(resources->flags);
  if (resources->sharpSettings->globalRank == resources->sharpSettings->nodeLeaderRank){
    free(resources->sharpSettings->reduce_spec.sbuf_desc.iov.vector);
    free(resources->sharpSettings->reduce_spec.rbuf_desc.iov.vector);    
    for(int r = 0; r < resources->sharpSettings->nodeCommSize; r++){
      if (SHARP_COLL_SUCCESS != sharp_coll_dereg_mr(resources->sharpSettings->sharpCtx, resources->sharpSettings->mrs[r])) {
	WARN("Sharp dereg mr failed");
	return ncclInternalError;	
      }
      CUDACHECK(cudaFree(resources->nodeRanksBuf[r].remPtr));
    }
    #if 1
    if (resources->sharpSettings->sharpComm != NULL){
      if (SHARP_COLL_SUCCESS !=sharp_coll_comm_destroy(resources->sharpSettings->sharpComm))
	WARN("Sharp coll comm destroy failed");	
    }
    resources->sharpSettings->sharpComm = NULL;
    #endif

  }
  else{
    CUDACHECK(cudaIpcCloseMemHandle(resources->nodeRanksBuf[resources->sharpSettings->localRank].remPtr));
    sleep(3);
    }
  
  if (resources->sharpSettings->sharpCtx != NULL){
    if (SHARP_COLL_SUCCESS != sharp_coll_finalize(resources->sharpSettings->sharpCtx)){
      WARN("Sharp ctx finalize failed");
      return ncclInternalError;
    }
    resources->sharpSettings->sharpCtx = NULL;
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
